# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:22:15 2023
03/08: Dataframe created with standardised pcl. Now each pcl can go to NN for conclusion
03/08: 15:30: The flow works. Pending: (1) How to call the model without actually running it. (2) Getting it for all clusters. (3) Getting corresponding prob  
03/08: 18:30 Modified Pointnet to sort (1)
TODO: From DF_file-standardized, create a summary to give for each pcl_id, no of clusters, their sizes, corresponding NN conclusion, probabilities etc
05/08: Summary file created, prob as well as expectation added
10/08: Tried creating a video through Create_video routine. However does not work. Hence created another program

07/08: Added time stamp in df file clusters
25/08: Second level clustering implemented
    @author: LocalAdmin
"""


import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import ntpath
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from numpy import unique
import sys
import torch
from datetime import datetime
#from Pointnet_with_TFNET_size_reduced import PointNet
from Pointnet import PointNet
#from Pointnet import PointNet
from matplotlib.animation import FFMpegWriter
from shapely.geometry import Polygon
from scipy import spatial
import time
from scipy import stats
from sklearn.cluster import AffinityPropagation
plt.rcParams['animation.ffmpeg_path']= 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
pd.options.mode.chained_assignment = None # To disable pandas slice assignment warnings

persistence = 2 # Valid selection 1,2, 3,5
features = 4 # Valid selection 3,4,5
pointnet = PointNet()
#pointnet.load_state_dict(torch.load(r'C:\Girish\mmwave\new_development\Counting_ppl\models\save_F4P1_11.pth')) # Use the correct model based on persistence and features
#pointnet.load_state_dict(torch.load(r'C:\Girish\mmwave\new_development\Counting_ppl\F4P2_reduced\save_RF4P2_49.pth')) # Use the correct model based on persistence and features
#pointnet.load_state_dict(torch.load(r'C:\Girish\mmwave\new_development\Counting_ppl\long_run\save_F4P2_99.pth'))
#pointnet.load_state_dict(torch.load(r'C:\Girish\mmwave\new_development\Counting_ppl\models\save_F4P3_11.pth'))
pointnet.load_state_dict(torch.load(r'C:\Girish\mmwave\new_development\Counting_ppl\save_RF4P2_0.pth')) # final check
pointnet.eval()

#%%

# Function to standardize the cluster within a pointcloud. Each cluster )should have fixed dimensions. Each standardized cluster represents a single input to ML algorithm

class Standardize_cluster:
    
    def pad_df(self): # Is there any better way to pad
        df_col_list = list(self.curr_df.columns)
        
        #Padding with zeros -- option 1
        #zero_data = np.zeros(shape=(self.clustersize-self.curr_df.shape[0],len(df_col_list)))
        #df_aug = pd.DataFrame(zero_data, columns=df_col_list)
        #processed_df=pd.concat([self.curr_df,df_aug],ignore_index=True)
        
        # pad with highest snr values -- option 2
        #supp_data = self.curr_df.sort_values('snr',ascending=False).head(self.clustersize % self.curr_df.shape[0])
        #df_aug = pd.DataFrame(supp_data, columns=df_col_list)
        #processed_df = pd.DataFrame(columns=df_col_list)
        #while self.clustersize-processed_df.shape[0] >= self.curr_df.shape[0]:
        #    processed_df = pd.concat([processed_df,self.curr_df],ignore_index=True) # keep repeating the data integer no of times till possible
        #processed_df = pd.concat([processed_df,df_aug],ignore_index=True)
        
       # Pad with centroids -- option 3 
        supp_data_size = self.clustersize % self.curr_df.shape[0]
        df_aug = pd.DataFrame(columns=df_col_list)
        processed_df = pd.DataFrame(columns=df_col_list)
        if (supp_data_size != 0):
            cen = np.empty([supp_data_size,2]) 
            snr = np.empty(supp_data_size) 
            dopp = np.empty(supp_data_size) 
            seq = np.empty(supp_data_size)
            
            col1 = self.curr_df.y.to_numpy()
            col2 = self.curr_df.x.to_numpy()
            col3 = self.curr_df.snr.to_numpy()
            col4 = self.curr_df.dopp.to_numpy()
            col5 = self.curr_df.seq.to_numpy()
            aggloclust = AgglomerativeClustering(n_clusters=supp_data_size).fit(col1.reshape(-1,1),col2.reshape(-1,1))
            labels = aggloclust.labels_
            
            for i in range (0,supp_data_size):
                sel_y = [col1[j] for j in range(len(col1)) if labels[j] == i]
                sel_x = [col2[j] for j in range(len(col2)) if labels[j] == i]
                sel_snr = [col3[j] for j in range(len(col3)) if labels[j] == i]
                sel_dopp = [col4[j] for j in range(len(col4)) if labels[j] == i]
                sel_seq = [col5[j] for j in range(len(col5)) if labels[j] == i]
                cen[i] = np.average([sel_y,sel_x],axis=1,weights=sel_snr)
                snr[i] = np.average(sel_snr)
                dopp[i] = np.average(sel_dopp)
                seq[i] = stats.mode(sel_seq,keepdims=False).mode    
            
            df_aug['y'] = cen[:,0]
            df_aug['x'] = cen[:,1]
            df_aug['snr'] = snr
            df_aug['dopp'] = dopp
            df_aug['seq'] = seq
        while self.clustersize-processed_df.shape[0] >= self.curr_df.shape[0]:
            processed_df = pd.concat([processed_df,self.curr_df],ignore_index=True) # keep repeating the data integer no of times till possible
        processed_df = pd.concat([processed_df,df_aug],ignore_index=True)
        return(processed_df)
    
    
    def trim_df(self):
        processed_df = self.curr_df.sort_values('snr',ascending=False).head(self.clustersize) # If required to trim, trim out the lowest snr values
        processed_df = processed_df.sort_index(axis=0,ascending=True)
        return(processed_df)

    def standardize_df(self):
        if(self.curr_df.shape[0] < self.clustersize):
            self.processed_df = self.pad_df()
        elif(self.curr_df.shape[0] > self.clustersize):
            self.processed_df = self.trim_df()        
        else:    
            self.processed_df = self.curr_df
        self.paddedsize = self.clustersize - self.curr_df.shape[0] # positive when padded, negative when trimmed

    def __init__(self, cluster, features, persistence): # dataframe for a single cluster within a file
        self.paddedsize = 0
        self.clustersize = 0
        if persistence == 1:
            self.clustersize = 128
        elif persistence == 2:
            self.clustersize = 256
        elif persistence == 3:
            self.clustersize = 512
        elif persistence == 5:
            self.clustersize = 1024
        else:
            print("Invalid persistence ... Exiting")
            sys.exit()
            
        self.df_col_list = ['range','azim','dopp','snr','y','x','current_frame','seq','frm_cluster']
        if features ==3:
            self.selected_cols_for_csv=['y','x','seq']
        elif features == 4:
            self.selected_cols_for_csv=['y','x','seq','snr']
        elif features == 5:
            self.selected_cols_for_csv=['y','x','seq','snr','dopp']
        else:
            print("Invalid selection for features... Exiting")
            sys.exit()
        
        self.curr_df = cluster
        self.standardize_df()

#%%
        
class GetPtcld:
    # Extract Frames
    def extract_frames(self):
        filename = self.f
        with open(filename) as csv_file:
            reader = csv.reader(csv_file, delimiter=",", strict=False)
            single_frame = []
            frames=[]
            frame_cnt = 0
            for row in reader:
                if row[0] == "M":
                    frames.append(single_frame)
                    single_frame= []
                    frame_cnt +=1
                single_frame.append(row)
        frames.append(single_frame) # appending the last frame. Total size of frames is 1000+1, first being header
        return(frames,frame_cnt)
 
    def path_leaf(self,path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    # Polar to cartesian point cloud
    def polar2cart(self,ptcld, current_frame, ts):
        ptcld_cart=np.empty((len(ptcld),8))
        for j in range(len(ptcld)):
            ptcld_cart[j][0]=float(ptcld[j][0])  #range
            ptcld_cart[j][1]=float(ptcld[j][1])  #azim
            ptcld_cart[j][2]=float(ptcld[j][2]) #*math.cos(ptcld_cart[j][1])  #doppler -- Check
            ptcld_cart[j][3]=float(ptcld[j][3])  #snr
            ptcld_cart[j][4]=ptcld_cart[j][0]*math.cos(ptcld_cart[j][1]) #y
            ptcld_cart[j][5]=ptcld_cart[j][0]*math.sin(ptcld_cart[j][1]) #x
            ptcld_cart[j][6]=current_frame
            ptcld_cart[j][7]=ts
        return(ptcld_cart)
    
    
    # Extract point clouds from each frame
    def extract_ptcld(self,single_frame):
        ptcld=[]
        for i in range(len(single_frame)): # Going through each row of frame
            if single_frame[i][0]=="range": # Start of pt cld information
                ptcld=[]
            elif single_frame[i][0]=="TID":
                break
            else:
                ptcld.append(single_frame[i])
        return(ptcld)
    
    
    def plot_filter_ptcld_persistence(self,file_ptclds,file_fn,file_ts): # commented lines could be used for plotting the pointclouds
        file_filtered_ptclds=[]
        ptclds_pointer=[]
        #plt.ion()
        #fig=plt.figure()
        #ax = fig.add_subplot(111)
        persistence = self.persistence
        filename = self.f

        for i in range(len(file_ptclds)-persistence): #no o ptclds
            pts_to_be_plotted=[]    
            #default_pt = file_ptclds[0][0] # default pt containing only frame no and ts
            #default_pt = [-1,-1,-1,-1,-1,-1,file_fn[i],file_ts[i]]
            for m in range(0,persistence):
                for j in range(len(file_ptclds[i+m])): #no of pts
                    pt = file_ptclds[i+m][j]
                    #pt[7]=m # start sequencing with 0
                    pt = np.append(pt,m) #sequencing information. Gives information about persistence
                    pt = np.append(pt,i) # distinct identification for each pointcloud. Due to persistence, the frame no cant be the id.
                    if(pt[5]>self.minx and pt[5]<self.maxx and pt[4]>self.miny and pt[4]<self.maxy and pt[3]>self.snr_thresh and pt[2]>self.minv and pt[2]<self.maxv): # condition plotting static points
                            pts_to_be_plotted.append(pt) 
            if pts_to_be_plotted != []: # Check that there is atleast one point to plot
                file_filtered_ptclds.append(pts_to_be_plotted)
            ptclds_pointer.append([i,file_fn[i],file_ts[i]])
            #else:
                #file_filtered_ptclds.append(default_pt)
                #y = [item[4] for item in pts_to_be_plotted]
                #x = [item[5] for item in pts_to_be_plotted]
                #snr = [item[3] for item in pts_to_be_plotted]
            #current_frame = file_fn[i]
            #current_ts = file_ts[i]
            #print(current_frame + "\n")
            #ax.scatter(x,y,alpha=0.8,c=snr, cmap='YlOrBr',s=10)
 
            #plt.title(self.path_leaf(filename) + "    " + "Frame "+ current_frame + "     " + "UTC Time "+ current_ts + "     " )
            #plt.grid()
            #ax.set_xlim(self.minx,self.maxx)
            #ax.set_ylim(self.miny,self.maxy)
            #ax.set_xlabel("X")
            #ax.set_ylabel("Y")
            #plt.draw()
            #plt.pause(0.01)
            #ax.cla()
        #plt.close()
        #print(file_filtered_ptclds)
        return(file_filtered_ptclds,ptclds_pointer)


    def single_file_plot(self):
        frames,frame_cnt = self.extract_frames()
        file_ptclds = [] # all point clouds in a file
        file_fn = []
        file_ts = []

        try:
            for i in range(1,frame_cnt+1):
                ptcld = self.extract_ptcld(frames[i])
                ptcld_cart=np.empty((len(ptcld),6))
                if(frames[i][1][4]) !=[]:
                    current_frame=frames[i][1][5] #frame no
                    ts=int(float(frames[i][1][10])) #time
                    current_time = datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')
                if ptcld != []:
                    #ptcld_cart = self.polar2cart(ptcld,current_frame)    # Temporary change
                    ptcld_cart = self.polar2cart(ptcld, current_frame, ts)    
                    file_ptclds.append(ptcld_cart)
                    file_fn.append(current_frame)
                    file_ts.append(ts)
                #else:
                    #file_ptclds.append([-1,-1,-1,-1,-1,-1,current_frame,ts]) # just append current frame and ts
            self.file_filtered_ptclds, self.file_ptclds_pointer = self.plot_filter_ptcld_persistence(file_ptclds,file_fn,file_ts)
        except KeyboardInterrupt:
            print("Key board interrupt...")

    
    
    def __init__(self, path, file, persistence):
        self.path = path  
        self.file = file
        self.snr_thresh=0
        self.minx=-10
        self.maxx=5
        self.miny=0.5
        self.maxy=12.5
        self.snr_thresh=0
        self.persistence = persistence
        self.plot_static = False
        self.minv = 0 #0 for dynamic, -ve for static
        self.maxv=100
        self.dest_directory = os.path.join(self.path, "processed")
        self.file_filtered_ptclds = []
        self.file_ptclds_pointer = []
        if not(os.path.exists(self.dest_directory)):
            os.mkdir(self.dest_directory)
        self.f = os.path.join(self.path, self.file)
        if os.path.isfile(self.f):
            print(self.f)
            self.single_file_plot()
            os.rename(self.f,os.path.join(self.dest_directory,self.file))
        else:
            print("Given file does not exist at the stated path")
            print("Checked for: ", str(self.f))
            sys.exit()


#%%

class CreateClusters:
    
    # Filter each frame further to check for minimum points in each frame before filtering
    def precluster_filtering(self):
        for i in range(len(self.ptclds)):
            if len(self.ptclds[i]) > self.minpts:
                self.pts_to_cluster.append(self.ptclds[i])

    
    
    def cluster_pts(self):
        df_col_list = ['range','azim','dopp','snr','y','x','current_frame','ts','seq','pcl_id']
        model = DBSCAN(eps=self.eps, min_samples= self.minsamples)
        self.precluster_filtering() 
        #self.pts_to_cluster = self.ptclds
        if len(self.pts_to_cluster) == 0:
            print("No points to cluster in this file after preprocessing")
        else:
            print(len(self.pts_to_cluster))
            for i in range(len(self.pts_to_cluster)): # one pcl_id at a time
                DF_frm = pd.DataFrame(self.pts_to_cluster[i],columns= df_col_list)
                DF_num = DF_frm[['x','y']].to_numpy()
                frm_cluster = model.fit_predict(DF_num) # create clusters using x and y
                DF_frm['frm_cluster'] = frm_cluster # Creating clusters for each pointcloud (across frames depending upon persistence)
                self.DF_file = pd.concat([self.DF_file,DF_frm], ignore_index=True)

                
   
    def __init__(self, ptclds):
        self.minpts = 20 # minimum number of pts in a frame to process further for clustering
        self.minsamples = 10 # min_samples for DBSCAN
        self.eps = 0.5 # eps for DBSCAN
        self.ptclds = ptclds
        self.DF_file = pd.DataFrame()
        self.pts_to_cluster = [] # after filtering the ptclds for minimum total pts
        self.cluster_pts()


        
        

#%% Other functions

## Use this routine only to visualize the clusters for study or to store the images. There is a different program to create video
def visualize_cluster(DF_file_clusters, pcl_id, fno, fts, fexp, save_images, img_directory):
    
    frm_cluster_values = [-1,0,1,2,3,4,5,6,7,8,9]
    colors = ['k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan','tab:gray']
    colordict = dict(zip(frm_cluster_values,colors))
    
    pcl_to_viz = DF_file_clusters.loc[DF_file_clusters['pcl_id'] == pcl_id]
    pcl_to_viz_data = pcl_to_viz[["x", "y", "frm_cluster"]].dropna(how="any")
    pcl_to_viz_data["color"] = pcl_to_viz_data["frm_cluster"].apply(lambda x: colordict[x])
    vals = pcl_to_viz_data.values
    
    #cmap = 'Spectral'
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(vals[:, 0], vals[:, 1],c=vals[:, 3],label=vals[:, 3])
    
    plt.title("PCL_ID "+ str(pcl_id) + " Frame_No:" +str(fno) +" TS:"+ datetime.utcfromtimestamp(fts).strftime('%H:%M:%S') + " Expected_no:" + str(round(fexp,2)))
    plt.grid()
    ax.set_xlim(-10, 5)
    ax.set_ylim(0.5, 15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    #plt.legend()

    img_path = os.path.join(img_directory,str(pcl_id) + ".png")
    if save_images:
        plt.savefig(img_path)
        plt.close()
    else:
        plt.draw()
        plt.pause(0.5)
        input("Press Enter to continue...")
        #plt.pause(0.5)
        ax.cla()
        plt.close()

#%%
# =============================================================================
# def plot_empty(curr_pclid, fig, ax, curr_ts, lapse):
#     X=[]
#     Y=[]
#     ax.scatter(X,Y)
#     plt.grid()
#     plt.title("Frame:"+ str(curr_pclid) + " TS:" + datetime.utcfromtimestamp(curr_ts).strftime('%H:%M:%S') )
#     ax.set_xlim(-10, 5)
#     ax.set_ylim(0.5, 15)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     #plt.pause(lapse)
# 
# =============================================================================

#%% Get cluster area
def Get_cluster_area(clstr):
    X_coor = clstr.x.to_numpy()
    Y_coor = clstr.y.to_numpy()
    points = np.vstack((X_coor,Y_coor)).T
    hull = spatial.ConvexHull(points)
    #for simplex in hull.simplices:
        #plt.plot(points[simplex, 0], points[simplex, 1], 'r-')
    #print("Area is",hull.volume)    
    return(hull.volume)

#%% Give the expected value of people in a cluster based on output class probabilities. Bike is considered as one person and bigger is considered as 5 ppl
def Get_exp_value(outputs):
    pr_op = torch.exp(outputs)
    pr_np = pr_op.detach().numpy()
    pr_np = np.squeeze(pr_np)
    exp_value = 1*pr_np[0]+2*pr_np[1]+3*pr_np[2]+4*pr_np[3]+5*pr_np[4]+1*pr_np[5]
    return(exp_value)

#%% Give the clusters one by one for interpretation
def Get_cluster_id(pcl):
    sdf = pcl
    sel_col = ['y','x','seq','snr']
    pcl_np = sdf[sel_col].to_numpy()
    b = torch.from_numpy(pcl_np).float()
    pcl_t = b.unsqueeze(0)
    outputs, __, __ = pointnet(pcl_t.transpose(1,2))
    exp_no = Get_exp_value(outputs)
    #pred = torch.max(outputs.data, 1)
    prob,pred = torch.max(outputs.data, 1) # prob is the probability of the most probable class
    #print(pred.item(),np.exp(prob.item()))
    return(pred.item(),np.exp(prob.item()),exp_no)
    
#%% Recluster when the size of cluster is higher
def Recluster(cluster_to_process):
    recluster_preference = -60
    reclustered = cluster_to_process.copy()
    reclustered.drop(labels=['frm_cluster'], axis="columns", inplace=True)
    X = np.vstack((reclustered.x.to_numpy(),reclustered.y.to_numpy())).T
    af = AffinityPropagation(preference=recluster_preference, random_state=0,damping=0.75,convergence_iter=50).fit(X)
    labels = af.labels_+100 # 100 added just to differentiate from first set of labels
    reclustered['frm_cluster'] = labels
    return(reclustered)

#%% Append the count information to DF file header
def append_DF_file_header(DF_file_summary,DF_file_header):
    for i in range(len(DF_file_header)):
        #print(DF_file_header.pcl_id[i])
        e = DF_file_summary.loc[DF_file_summary['pcl_id'] == int(DF_file_header.pcl_id[i])] 
        if e.empty == False:
            #print(e)
            cnt = e.exp_no.sum()
            print(i,cnt)
        else:
            cnt = 0
        DF_file_header.at[i,'count'] = cnt
    return (DF_file_header)
#%%
# File data
#path = r"C:\Girish\mmwave\new_development\AMS_Deployment\Initial\2908\mmwave"
if __name__ == '__main__':
    path = r"C:\Girish\mmwave\new_development\AMS_Deployment\Initial\0809\processed"
    #path = r"C:\Girish\mmwave\new_development\AMS_Deployment\Initial\2007\raw"
    df_directory = os.path.join(path, "df")
    img_directory = os.path.join(path, "outputs") # folder for storing the images
    #file = "Text_23May143829.csv"
    #file = "Text_08Jun153423.csv"
    #file = "Text_05Sep110643.csv" #example
    #file = "Text_20Jul000638.csv" #car
    #file = "Text_20Jul002504.csv" #dir debug
    file = "Text_08Sep190628.csv" #good crowd
    
    enable_sec_clustering = True
    sec_clustering_thresh = 12
    DF_file_processed = pd.DataFrame() # Dataframe containing the processed clusters for the entire file with ops from NN   
    DF_file_summary = pd.DataFrame(columns=['pcl_id','frames','frm_cluster','cluster_size','cluster_label','cluster_prob','exp_no']) # Dataframe containing the summary for the entire file
    #Df_file_comp = pd.DataFrame(columns=['pcl_id','frame','ts','count'])
    save_images = False   # True for saving images, false for displaying images
    if not(os.path.exists(df_directory)):
        os.mkdir(df_directory)
    if not(os.path.exists(img_directory)):
        os.mkdir(img_directory)
    
    # Extract point clouds
    ptclds = GetPtcld(path, file, persistence)
    
    # If point clouds are present, create clusters
    if ptclds.file_filtered_ptclds !=[]:
        clusters = CreateClusters(ptclds.file_filtered_ptclds) 
        DF_file_clusters = clusters.DF_file # clusters.DF_file gives a dataframe of all pts in the file after filtering and clustering for each frame
        DF_file_header = pd.DataFrame(np.row_stack(ptclds.file_ptclds_pointer),columns=['pcl_id','frame','ts']) # DF containing only the framenos and timestamps along with pcl_id (including empty ones)
    else:
        print("No point clouds to process in this file")
        sys.exit()
    
    if (DF_file_clusters.empty == False):
        pcls = unique(DF_file_clusters.pcl_id) # pcls contain the list of all pointclouds in the file
        for pcl in pcls: # one pointcloud at a time
            pcl_to_process = DF_file_clusters.loc[DF_file_clusters['pcl_id'] == pcl] #This gives the pcl data (across persistence) at one snap
            #frm_to_process.index = pd.RangeIndex(len(frm_to_process.index))
            
            
            if enable_sec_clustering:
                clusters_in_pcl = unique(pcl_to_process.frm_cluster)
                for cluster in clusters_in_pcl: # one cluster at a time within the pclx`
                    if cluster != -1: # Process only if the given cluster does not indicate noise, 
                        cluster_to_process = pcl_to_process.loc[pcl_to_process['frm_cluster'] == cluster].copy()
                        cluster_to_process_idx = pcl_to_process.index[pcl_to_process['frm_cluster'] == cluster].tolist()
                        clstr_area = Get_cluster_area(cluster_to_process)
                        if (clstr_area > sec_clustering_thresh):
                            reclustered = Recluster(cluster_to_process) # recluster the bigger clusters
                            pcl_to_process.loc[cluster_to_process_idx] = reclustered # and change the original with the reclustered ones
               
            clusters_in_pcl = unique(pcl_to_process.frm_cluster)        
            for cluster in clusters_in_pcl: # one cluster at a time within the pcl
                if cluster != -1: # Process only if the given cluster does not indicate noise, 
                    cluster_to_process = pcl_to_process.loc[pcl_to_process['frm_cluster'] == cluster].copy()
                    clstr_area = Get_cluster_area(cluster_to_process)
                    cluster_to_process_std = Standardize_cluster(cluster_to_process,features, persistence) #this output can go as input for ML. This is just one cluster within one frame from a single file
                    cid, prob, exp_no = Get_cluster_id(cluster_to_process_std.processed_df) # Output from NN to get the class corresponding to this cluster
                    #a = cluster_to_process_std.processed_df
                    a = cluster_to_process
                    a['cluster_label'] = cid
                    a['cluster_size'] = cluster_to_process.shape[0]
                    a['cluster_prob'] = prob
                    a['exp_no'] = exp_no
                    a['cluster_area'] = clstr_area
                    DF_file_processed = pd.concat([DF_file_processed,a], ignore_index=True)
                    b=pd.DataFrame([[pcl, unique((pcl_to_process.current_frame)),cluster,cluster_to_process.shape[0],cid,prob,exp_no]],columns=['pcl_id','frames','frm_cluster','cluster_size','cluster_label','cluster_prob','exp_no'])
                    DF_file_summary = pd.concat([DF_file_summary,b], ignore_index=True)
                    print("PCL_id, cluster, Size, Cluster label, Prob, Exp_no",pcl, cluster,cluster_to_process.shape[0],cid, round(prob,2), round(exp_no,2))
            fno = int(cluster_to_process.current_frame.head(1).item()) #Retrieving frame no from the pcl   
            fts = int(cluster_to_process.ts.head(1).item()) #Retrieving timestamp from the pcl    
            summ_curr_pcl = DF_file_summary.loc[DF_file_summary['pcl_id'] == pcl]
            fexp = summ_curr_pcl['exp_no'].sum() # Gives the expected no for the entire pcl
            #visualize_cluster(DF_file_clusters,pcl,fno,fts,fexp,save_images,img_directory)
        ## Copy count information in summary file
        DF_file_header = append_DF_file_header(DF_file_summary,DF_file_header)
        DF_file_summary.to_csv(os.path.join(img_directory,"DF_file_summary.csv"))
        DF_file_header.to_csv(os.path.join(img_directory,"DF_file_header.csv"))
        DF_file_processed.to_csv(os.path.join(img_directory,"DF_file_processed.csv"))
    else:
        print("Pointcloud present but size was not sufficient")

#%% Example for standardization

# =============================================================================
# frm_to_process = DF_file_clusters.loc[DF_file_clusters['current_frame'] == 101373]
# frm_to_process.index = pd.RangeIndex(len(frm_to_process.index))
# clusters_in_frm = unique(frm_to_process.frm_cluster)
# for cluster in clusters_in_frm: # one cluster at a time within the frame
#     cluster_to_process = frm_to_process.loc[frm_to_process['frm_cluster'] == cluster]
#     a = Standardize_cluster(cluster_to_process,4,5) #a can go as input for ML. This is just one cluster within one frame from a single file
# #%%
# 
# # Example code for getting class for single cluster
# df_col_list = ['range','azim','dopp','snr','y','x','current_frame','seq']
# #fn = r'C:\Girish\mmwave\Training_data\Training_data_P5\1\test\Text_01Jun141600.csv85777.0.csv'
# fn = r'C:\Girish\mmwave\Training_data\Training_data_P3\1\test\Text_01Jun141600.csv85791.0.csv'
# #fn = r'C:\Girish\mmwave\Training_data\Training_data_P5\bigger\test\Text_09Jun114839.csv862159.0.csv'
# sdf = pd.read_csv(fn,names= df_col_list)
# sel_col = ['y','x','seq','snr']
# pcl = sdf[sel_col].to_numpy()
# b = torch.from_numpy(pcl).float()
# #print(b)
# d=b.unsqueeze(0)
# outputs, __, __ = pointnet(d.transpose(1,2))
# _,pred = torch.max(outputs.data, 1)
# print(pred.item())
# =============================================================================
#%%

