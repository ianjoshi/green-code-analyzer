import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split

def analyze_model_fairness(data_path, model_paths, protected_attributes, with_plot=False):
    data = pd.read_csv(data_path)
    data = data.astype(np.float32)
    data = data.drop(["Ja", "Nee"], axis=1)
    
    if with_plot:
        for attribute in protected_attributes:
            plt.figure()
            plt.title(attribute)
            plt.hist(data[attribute])
            plt.show()
            print(f"{attribute}: {data[attribute].unique()}, max={data[attribute].unique().max()}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create privileged and unprivileged groups for each protected attribute
    for attribute in protected_attributes:
        if attribute == 'persoon_leeftijd_bij_onderzoek':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] > 40
        elif attribute == 'adres_aantal_brp_adres':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] < 4
        elif attribute == 'adres_aantal_verschillende_wijken':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] < 3
        elif attribute == 'adres_aantal_verzendadres':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] < 1
        elif attribute == 'adres_aantal_woonadres_handmatig':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] < 1
        elif attribute == 'adres_dagen_op_adres':
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] > 6000
        else:
            test_data[f'privileged_group_{attribute}'] = test_data[attribute] == 0

    # Prepare dataset for AIF360
    protected_attribute_names = [f'privileged_group_{attribute}' for attribute in protected_attributes]

    results = {}

    for model_path in model_paths:
        model_name = model_path.split('/')[-1].split('.')[0]
        session = ort.InferenceSession(model_path)

        # Get model predictions
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: test_data.drop(['checked'] + protected_attribute_names, axis=1).values.astype(np.float32)})[0]

        # Create dataset with predictions
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=test_data,
            label_names=['checked'],
            protected_attribute_names=protected_attribute_names
        )

        pred_dataset = dataset.copy()
        pred_dataset.scores = predictions

        feature_results = {}

        # Calculate fairness metrics for each protected attribute
        for attribute in protected_attribute_names:
            metrics = ClassificationMetric(
                dataset,
                pred_dataset,
                unprivileged_groups=[{attribute: 0}],
                privileged_groups=[{attribute: 1}]
            )

            feature_results[attribute] = {
                'disparate_impact': metrics.disparate_impact(),
                'statistical_parity_difference': metrics.statistical_parity_difference(),
                'equal_opportunity_difference': metrics.equal_opportunity_difference(),
                'average_odds_difference': metrics.average_odds_difference()
            }

        results[model_name] = feature_results

    return results

def fairness_test(X_test : pd.DataFrame, Y_test : pd.Series, session, protected_attributes = None):
    if protected_attributes is None:
        protected_attributes = [
            'persoon_leeftijd_bij_onderzoek',
            'adres_aantal_brp_adres',
            'adres_aantal_verschillende_wijken',
            'adres_aantal_verzendadres',
            'adres_aantal_woonadres_handmatig',
            'adres_dagen_op_adres',
            'adres_recentst_onderdeel_rdam',
            'adres_recentste_buurt_groot_ijsselmonde',
            'adres_recentste_buurt_nieuwe_westen',
            'adres_recentste_buurt_other',
            'adres_recentste_buurt_oude_noorden',
            'adres_recentste_buurt_vreewijk',
            'adres_recentste_plaats_other',
            'adres_recentste_plaats_rotterdam',
            'adres_recentste_wijk_charlois',
            'adres_recentste_wijk_delfshaven',
            'adres_recentste_wijk_feijenoord',
            'adres_recentste_wijk_ijsselmonde',
            'adres_recentste_wijk_kralingen_c',
            'adres_recentste_wijk_noord',
            'adres_recentste_wijk_other',
            'adres_recentste_wijk_prins_alexa',
            'adres_recentste_wijk_stadscentru',
            'adres_unieke_wijk_ratio'
        ]
        
    # Create privileged and unprivileged groups for each protected attribute
    for attribute in protected_attributes:
        if attribute == 'persoon_leeftijd_bij_onderzoek':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] > 40
        elif attribute == 'adres_aantal_brp_adres':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] < 4
        elif attribute == 'adres_aantal_verschillende_wijken':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] < 3
        elif attribute == 'adres_aantal_verzendadres':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] < 1
        elif attribute == 'adres_aantal_woonadres_handmatig':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] < 1
        elif attribute == 'adres_dagen_op_adres':
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] > 6000
        else:
            X_test[f'privileged_group_{attribute}'] = X_test[attribute] == 0

    # Prepare dataset for AIF360
    protected_attribute_names = [f'privileged_group_{attribute}' for attribute in protected_attributes]

    # Get model predictions
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: X_test.drop(protected_attribute_names, axis=1).values.astype(np.float32)})[0]

    # Create DataFrame with features and label
    test_data = pd.concat([X_test, Y_test.rename('checked')], axis=1)
    
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=test_data,
        label_names=['checked'],
        protected_attribute_names=protected_attribute_names
    )

    pred_dataset = dataset.copy()
    pred_dataset.scores = predictions

    feature_results = {}

    # Calculate fairness metrics for each protected attribute
    for attribute in protected_attribute_names:
        metrics = ClassificationMetric(
            dataset,
            pred_dataset,
            unprivileged_groups=[{attribute: 0}],
            privileged_groups=[{attribute: 1}]
        )

        feature_results[attribute] = {
            'disparate_impact': metrics.disparate_impact(),
            'statistical_parity_difference': metrics.statistical_parity_difference(),
            'equal_opportunity_difference': metrics.equal_opportunity_difference(),
            'average_odds_difference': metrics.average_odds_difference()
        }

    return feature_results


def print_fairness_report(results):
    for model_name, features in results.items():
        print(f"\n=== Fairness Report for {model_name} ===")

        for feature, metrics in features.items():
            print(f"\nFairness Metrics for {feature}:")
            print(f"Disparate Impact: {metrics['disparate_impact']:.3f}")
            print(f"Statistical Parity Difference: {metrics['statistical_parity_difference']:.3f}")
            print(f"Equal Opportunity Difference: {metrics['equal_opportunity_difference']:.3f}")
            print(f"Average Odds Difference: {metrics['average_odds_difference']:.3f}")


def plot_fairness_results(results, save_path='results/comparison_results/fairness_plot.png'):
    clean_results = {}
    for model_name in results:
        clean_name = f"model{model_name.split('model_')[1]}"
        clean_results[clean_name] = results[model_name]

    results = clean_results

    metrics = ['disparate_impact', 'statistical_parity_difference',
               'equal_opportunity_difference', 'average_odds_difference']

    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 12

    for metric in metrics:
        plt.figure()

        # Prepare data for plotting
        models = list(results.keys())
        features = list(results[models[0]].keys())
        x = np.arange(len(features))
        width = 0.35

        colors = ['#2196F3', '#4CAF50']  # Blue and Green
        for i, (model, color) in enumerate(zip(models, colors)):
            metric_values = [results[model][feature][metric] for feature in features]
            plt.bar(x + i * width, metric_values, width, label=model,
                    alpha=0.8, color=color)

        plt.xlabel('Protected Attributes', fontsize=12, labelpad=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, labelpad=10)
        plt.title(f'{metric.replace("_", " ").title()} by Protected Attribute',
                  fontsize=14, pad=20)

        plt.xticks(x + width / 2,
                   [f.replace('privileged_group_', '').replace('_', ' ')
                    for f in features],
                   rotation=45, ha='right')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        if metric == 'disparate_impact':
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.3,
                        label='Reference (y=1)')
        else:
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3,
                        label='Reference (y=0)')

        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(f"{save_path}_{metric}.png")

    plt.show()


# if __name__ == "__main__":
#     data_path = 'data\investigation_train_large_checked.csv'
#     model_paths = [
#         'models\model_1.onnx',
#         'models\model_2.onnx'
#     ]

#     protected_attributes = [
#         'persoon_leeftijd_bij_onderzoek',
#         'adres_aantal_brp_adres',
#         'adres_aantal_verschillende_wijken',
#         'adres_aantal_verzendadres',
#         'adres_aantal_woonadres_handmatig',
#         'adres_dagen_op_adres',
#         'adres_recentst_onderdeel_rdam',
#         'adres_recentste_buurt_groot_ijsselmonde',
#         'adres_recentste_buurt_nieuwe_westen',
#         'adres_recentste_buurt_other',
#         'adres_recentste_buurt_oude_noorden',
#         'adres_recentste_buurt_vreewijk',
#         'adres_recentste_plaats_other',
#         'adres_recentste_plaats_rotterdam',
#         'adres_recentste_wijk_charlois',
#         'adres_recentste_wijk_delfshaven',
#         'adres_recentste_wijk_feijenoord',
#         'adres_recentste_wijk_ijsselmonde',
#         'adres_recentste_wijk_kralingen_c',
#         'adres_recentste_wijk_noord',
#         'adres_recentste_wijk_other',
#         'adres_recentste_wijk_prins_alexa',
#         'adres_recentste_wijk_stadscentru',
#         'adres_unieke_wijk_ratio'
#     ]

#     results = analyze_model_fairness(data_path, model_paths, protected_attributes)
#     # plot_fairness_metrics(results)
#     print_fairness_report(results)
