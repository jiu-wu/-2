import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
from scipy.stats import mode

# 设置随机种子
RANDOM_STATE = 42


# 1. 创建必要的目录
def create_directories():
    directories = ['data', 'images', 'reports', 'logs', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    # 创建目录后再配置日志
    configure_logging()
    logging.info("必要的目录已创建或已存在。")


# 2. 配置日志
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/experiment.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("日志记录已配置。")


# 3. 数据加载与预处理

def load_data(train_path, test_path):
    try:
        train = pd.read_csv(train_path, sep='\s+', header=None, engine='python')
        test = pd.read_csv(test_path, sep='\s+', header=None, engine='python')
        logging.info("训练集和测试集数据加载成功。")
        return train, test
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        raise


def preprocess_data(train, test):
    # 合并训练和测试数据以进行统一预处理
    data = pd.concat([train, test], ignore_index=True)
    logging.info("训练集和测试集数据已合并。")

    # 替换标签为从0开始的整数
    original_labels = data.iloc[:, 0].unique()
    label_mapping = {label: idx for idx, label in enumerate(original_labels)}
    data.iloc[:, 0] = data.iloc[:, 0].map(label_mapping)
    logging.info(f"标签映射: {label_mapping}")

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"数据已拆分: 训练集大小={X_train.shape}, 测试集大小={X_test.shape}")
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    logging.info("数据标准化完成。")
    return X_train_std, X_test_std, scaler


# 4. 模型训练与评估

def train_supervised_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        logging.info(f"{name} 模型训练完成。")
    return trained_models


def evaluate_models(models, X_test, y_test):
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)
        metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Report': report,
            'Predictions': y_pred
        }
        logging.info(f"{name} 模型评估完成。准确性={acc:.4f}, 精确率={prec:.4f}, 召回率={rec:.4f}, F1分数={f1:.4f}")
    return metrics


def plot_confusion_matrix_custom(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"混淆矩阵已保存至 {save_path}")


def plot_metrics_comparison(metrics, save_path):
    df = pd.DataFrame({
        model: [
            metrics[model]['Accuracy'],
            metrics[model]['Precision'],
            metrics[model]['Recall'],
            metrics[model]['F1_Score']
        ]
        for model in metrics
    }, index=['Accuracy', 'Precision', 'Recall', 'F1_Score'])

    df = df.T
    df.plot(kind='bar', figsize=(12, 8))
    plt.title('模型性能比较')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"模型性能比较图已保存至 {save_path}")


# 5. 无监督学习：K均值聚类

def train_kmeans(X_train, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(X_train)
    logging.info("KMeans 聚类训练完成。")
    return kmeans


def evaluate_kmeans(kmeans, X_test, y_test):
    y_pred = kmeans.predict(X_test)
    # 使用多数投票的方法将簇标签映射到真实标签
    labels = np.zeros_like(y_pred)
    for i in range(kmeans.n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            labels[mask] = 0
        else:
            labels[mask] = mode(y_test[mask])[0]
    acc = accuracy_score(y_test, labels)
    prec = precision_score(y_test, labels, average='weighted', zero_division=0)
    rec = recall_score(y_test, labels, average='weighted', zero_division=0)
    f1 = f1_score(y_test, labels, average='weighted', zero_division=0)
    report = classification_report(y_test, labels, zero_division=0)
    logging.info(f"KMeans 聚类评估完成。准确性={acc:.4f}, 精确率={prec:.4f}, 召回率={rec:.4f}, F1分数={f1:.4f}")
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Report': report,
        'Predictions': labels
    }


# 6. 模型优化

def optimize_model(model_name, model, X_train, y_train):
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }
    elif model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2']
        }
    else:
        logging.warning(f"暂不支持对 {model_name} 进行优化。")
        return model

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"{model_name} 的最佳参数: {grid_search.best_params_}")
    logging.info(f"{model_name} 的最佳交叉验证准确性: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def plot_training_metrics(model, X_train, y_train, save_path):
    if isinstance(model, RandomForestClassifier):
        n_estimators = np.arange(10, 310, 10)
        train_scores = []
        for n in n_estimators:
            clf = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            train_scores.append(clf.score(X_train, y_train))
        plt.figure(figsize=(12, 8))
        plt.plot(n_estimators, train_scores, marker='o')
        plt.title('Random Forest 训练准确率随树的数量变化')
        plt.xlabel('树的数量')
        plt.ylabel('训练准确率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"随机森林训练准确率曲线已保存至 {save_path}")
    else:
        logging.warning("当前模型不支持绘制训练准确率曲线。")


# 7. 降维与特征选择

def apply_pca(X_train, X_test, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    logging.info(f"PCA 降维完成。保留的主成分数量: {X_train_pca.shape[1]}")
    return X_train_pca, X_test_pca, pca


def apply_feature_selection(X_train, y_train, X_test, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    logging.info(f"特征选择完成。选择的特征数量: {k}")
    return X_train_sel, X_test_sel, selector


# 8. 保存报告和模型

def save_classification_report(report, model_name, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} 分类报告\n")
        f.write(report)
    logging.info(f"分类报告已保存至 {save_path}")


def save_experiment_report(evaluation_results, best_model_name, optimized_results, report_path):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ECG5000 分类实验报告\n")
        f.write("=" * 50 + "\n\n")
        f.write("各模型评估指标:\n")
        for model, metrics in evaluation_results.items():
            f.write(f"--- {model} ---\n")
            f.write(f"准确性: {metrics['Accuracy']:.4f}\n")
            f.write(f"精确率: {metrics['Precision']:.4f}\n")
            f.write(f"召回率: {metrics['Recall']:.4f}\n")
            f.write(f"F1分数: {metrics['F1_Score']:.4f}\n")
            f.write("分类报告:\n")
            f.write(metrics['Report'] + "\n")
            f.write("-" * 50 + "\n")
        if optimized_results:
            f.write("\n优化后的模型评估:\n")
            for model, metrics in optimized_results.items():
                f.write(f"--- {model} ---\n")
                f.write(f"准确性: {metrics['Accuracy']:.4f}\n")
                f.write(f"精确率: {metrics['Precision']:.4f}\n")
                f.write(f"召回率: {metrics['Recall']:.4f}\n")
                f.write(f"F1分数: {metrics['F1_Score']:.4f}\n")
                f.write("分类报告:\n")
                f.write(metrics['Report'] + "\n")
                f.write("-" * 50 + "\n")
    logging.info(f"实验报告已保存至 {report_path}")


# 9. 主程序

def main():
    # 创建必要的目录并配置日志
    create_directories()

    # 路径设置
    train_path = 'data/ECG5000_TRAIN.txt'
    test_path = 'data/ECG5000_TEST.txt'

    # 检查数据文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("数据文件不存在，请确保 ECG5000_TRAIN.txt 和 ECG5000_TEST.txt 位于 data/ 目录下。")
        return

    # 加载和预处理数据
    try:
        train, test = load_data(train_path, test_path)
    except Exception as e:
        logging.error("加载数据失败。")
        return

    X, y = preprocess_data(train, test)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_std, X_test_std, scaler = standardize_data(X_train, X_test)

    # 训练监督学习模型
    supervised_models = train_supervised_models(X_train_std, y_train)
    supervised_metrics = evaluate_models(supervised_models, X_test_std, y_test)

    # 训练K均值聚类
    n_clusters = len(np.unique(y))
    kmeans = train_kmeans(X_train_std, n_clusters)
    kmeans_metrics = evaluate_kmeans(kmeans, X_test_std, y_test)

    # 汇总所有模型的指标
    all_metrics = supervised_metrics.copy()
    all_metrics['KMeans'] = kmeans_metrics

    # 打印并保存分类报告
    for model_name, metric in all_metrics.items():
        logging.info(f"=== {model_name} ===")
        logging.info(f"Accuracy: {metric['Accuracy']:.4f}")
        logging.info(f"Precision: {metric['Precision']:.4f}")
        logging.info(f"Recall: {metric['Recall']:.4f}")
        logging.info("Classification Report:")
        logging.info(metric['Report'])
        # 保存分类报告到文件
        report_path = f"reports/{model_name}_classification_report.txt"
        save_classification_report(metric['Report'], model_name, report_path)

    # 可视化各模型的性能比较
    plot_metrics_comparison(all_metrics, 'images/model_performance_comparison.png')

    # 可视化混淆矩阵（以最佳监督学习模型为例）
    best_model_name = max(supervised_metrics, key=lambda x: supervised_metrics[x]['Accuracy'])
    best_model = supervised_models[best_model_name]
    y_pred_best = supervised_metrics[best_model_name]['Predictions']
    plot_confusion_matrix_custom(
        y_test, y_pred_best,
        f'{best_model_name} 混淆矩阵',
        f'images/{best_model_name}_confusion_matrix.png'
    )

    # 可视化KMeans混淆矩阵
    plot_confusion_matrix_custom(
        y_test, kmeans_metrics['Predictions'],
        'KMeans 混淆矩阵',
        'images/KMeans_confusion_matrix.png'
    )

    # 选择表现最好的模型并优化
    logging.info(f"表现最好的模型是: {best_model_name}")

    optimized_results = {}
    optimized_model = None

    if best_model_name in ['Random Forest', 'SVM', 'Logistic Regression']:
        optimized_model = optimize_model(best_model_name, best_model, X_train_std, y_train)
        y_pred_optimized = optimized_model.predict(X_test_std)
        acc = accuracy_score(y_test, y_pred_optimized)
        prec = precision_score(y_test, y_pred_optimized, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_optimized, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_optimized, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred_optimized, zero_division=0)
        optimized_results[best_model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Report': report,
            'Predictions': y_pred_optimized
        }
        logging.info(
            f"优化后的 {best_model_name} 性能: 准确性={acc:.4f}, 精确率={prec:.4f}, 召回率={rec:.4f}, F1分数={f1:.4f}")
        # 保存分类报告
        save_classification_report(report, f'Optimized {best_model_name}',
                                   f"reports/Optimized_{best_model_name}_classification_report.txt")
        # 可视化混淆矩阵
        plot_confusion_matrix_custom(
            y_test, y_pred_optimized,
            f'优化后的 {best_model_name} 混淆矩阵',
            f'images/Optimized_{best_model_name}_confusion_matrix.png'
        )
        # 可视化训练过程
        training_metrics_path = f"images/{best_model_name}_training_metrics.png"
        plot_training_metrics(optimized_model, X_train_std, y_train, training_metrics_path)
        # 保存优化后的模型
        model_save_path = f"models/Optimized_{best_model_name}.pkl"
        joblib.dump(optimized_model, model_save_path)
        logging.info(f"优化后的模型已保存至 {model_save_path}")

    # 降维示例（PCA）并重新训练最佳模型
    X_train_pca, X_test_pca, pca = apply_pca(X_train_std, X_test_std, variance_threshold=0.95)
    # 选择重新训练的模型，可以是优化后的模型
    if optimized_model:
        model_to_train = optimized_model
        model_name_to_train = f"Optimized {best_model_name} with PCA"
    else:
        model_to_train = best_model
        model_name_to_train = f"{best_model_name} with PCA"

    # 重新训练模型
    model_to_train.fit(X_train_pca, y_train)
    y_pred_pca = model_to_train.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    prec_pca = precision_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    rec_pca = recall_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    f1_pca = f1_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    report_pca = classification_report(y_test, y_pred_pca, zero_division=0)
    logging.info(
        f"{model_name_to_train} 性能: 准确性={acc_pca:.4f}, 精确率={prec_pca:.4f}, 召回率={rec_pca:.4f}, F1分数={f1_pca:.4f}")
    # 保存分类报告
    save_classification_report(
        report_pca,
        model_name_to_train,
        f"reports/{model_name_to_train}_classification_report.txt"
    )
    # 可视化混淆矩阵
    plot_confusion_matrix_custom(
        y_test, y_pred_pca,
        f'{model_name_to_train} 混淆矩阵',
        f'images/{model_name_to_train}_confusion_matrix.png'
    )
    # 保存降维后的模型
    model_pca_save_path = f"models/{model_name_to_train}.pkl"
    joblib.dump(model_to_train, model_pca_save_path)
    logging.info(f"PCA 降维后的模型已保存至 {model_pca_save_path}")

    # 保存实验报告
    save_experiment_report(
        evaluation_results=all_metrics,
        best_model_name=best_model_name,
        optimized_results=optimized_results,
        report_path='reports/experiment_report.txt'
    )

    logging.info("整个实验流程已完成。")


if __name__ == "__main__":
    main()
