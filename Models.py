from ProcessData import *
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import  classification_report, r2_score

def AdaBoost_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    bdt.fit(X_train,y_train)
    y_prediction = bdt.predict(X_test)
    accuracy=np.mean(y_prediction == y_test)*100
    print('Mean Square Error For AdaBoost Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction) , 5))
    print("The achieved accuracy using Adaboost is " + str(round(accuracy , 3)))
    timer.toc()
    print('AdaBoost Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#----------------------------------------------------------------------------------------------------------------------------

def DecisionTree_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
    accuracy=np.mean(y_prediction == y_test)*100
    print('Mean Square Error For Decision Tree Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction) , 5))
    print("The achieved accuracy using Decision Tree is " + str(round(accuracy , 3)))
    timer.toc()
    print('Decision Tree Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------------------------
def LogisticRegression_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    cls = LogisticRegression()
    cls.fit(X_train,y_train)
    prediction= cls.predict(X_test)
    accuracy = np.mean(prediction == y_test)*100
    print('Mean Square Error For Logistic Regression : ', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('The achieved accuracy using Logistic Regression is  '+str(round(accuracy , 3)))
    timer.toc()
    print('Logistic Regression Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------------------------
def KNN_Model(X_train, X_test, y_train, y_test , K=40):
    timer = TicToc('timer')
    timer.tic()
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    accuracy = np.mean(prediction==y_test) * 100
    print('Mean Square Error For KNN Classifier : ', round(metrics.mean_squared_error(y_test, prediction) , 5))
    print("The achieved accuracy using KNN is " + str(round(accuracy , 3)))
    timer.toc()
    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#----------------------------------------------------------------------------------------------------------------------------

def KNN_Model_KTrials(X_train, X_test, y_train, y_test , K=40):
    timer = TicToc('timer')
    timer.tic()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    error = []
    accuracy=[]
    # Calculating error for K values between 1 and K
    for i in range(1, K):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        accuracy.append(np.mean(pred_i==y_test)*100)
    timer.toc()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), accuracy, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accurcy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Acc')
    plt.show()
    print('MSE Scores')
    print(error)
    print('Accurcy scores')
    print(accuracy)
    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#----------------------------------------------------------------------------------------------------------------------------
def SVM_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    svc = svm.SVC(kernel='poly', C=0.5 , degree=2).fit(X_train, y_train)
    predictions = svc.predict(X_test)
    accuracy = np.mean(predictions == y_test)*100
    print('Mean Square Error For SVM Classification : ', round(metrics.mean_squared_error(y_test, predictions) , 5))
    print('The achieved accuracy using SVM is  '+ str(round(accuracy , 3)))
    timer.toc()
    print('SVM Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
  
#----------------------------------------------------------------------------------------------------------------------------
def Kmean_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train,y_train)
    predict = kmeans.predict(X_test)
    accuracy = np.mean(predict == y_test)*100
    print('Mean Square Error For K-mean Classification : ',round(metrics.mean_squared_error(y_test, predict) , 5))
    print('The achieved accuracy using K-mean is  ' + str(round(accuracy , 3)))
    timer.toc()
    print('AdaBoost Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
#----------------------------------------------------------------------------------------------------------------------------
#PCA
def AdaBoost_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    sc = StandardScaler()
    X_train1 = sc.fit_transform(X_train_PCA)
    X_test1 = sc.transform(X_test1_PCA)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
    scaler = StandardScaler()
    scaler.fit(X_train1)
    X_train1 = scaler.transform(X_train1)
    X_test1 = scaler.transform(X_test1)
    bdt.fit(X_train1, y_train)
    y_prediction = bdt.predict(X_test1)
    accuracy = np.mean(y_prediction == y_test) * 100
    print('Mean Square Error For AdaBoost Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction), 5))
    print("The achieved accuracy using Adaboost is " + str(round(accuracy, 3)))
    timer.toc()
    print('AdaBoost Classifier Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')


# ----------------------------------------------------------------------------------------------------------------------------

def DecisionTree_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train_PCA, y_train)
    y_prediction = clf.predict(X_test1_PCA)
    accuracy = np.mean(y_prediction == y_test) * 100
    print('Mean Square Error For Decision Tree Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction), 5))
    print("The achieved accuracy using Decision Tree is " + str(round(accuracy, 3)))
    timer.toc()
    print('Decision Tree Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')


# -----------------------------------------------------------------------------------------------------------------------------
def LogisticRegression_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_PCA)
    X_test = sc.transform(X_test1_PCA)
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    prediction = cls.predict(X_test)
    accuracy = np.mean(prediction == y_test) * 100
    print('Mean Square Error For Logistic Regression : ', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('The achieved accuracy using Logistic Regression is  ' + str(round(accuracy, 3)))
    timer.toc()
    print('Logistic Regression Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')


# -----------------------------------------------------------------------------------------------------------------------------
def KNN_Model_PCA(X_train, X_test, y_train, y_test, K=40):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_PCA, y_train)
    prediction = knn.predict(X_test1_PCA)
    accuracy = np.mean(prediction == y_test) * 100
    print('Mean Square Error For KNN Classifier : ', round(metrics.mean_squared_error(y_test, prediction), 5))
    print("The achieved accuracy using KNN is " + str(round(accuracy, 3)))
    timer.toc()
    print('KNN Classifier Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')

# ----------------------------------------------------------------------------------------------------------------------------
def SVM_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    svc = svm.SVC(kernel='poly', C=0.5, degree=2).fit(X_train_PCA, y_train)
    predictions = svc.predict(X_test1_PCA)
    accuracy = np.mean(predictions == y_test) * 100
    print('Mean Square Error For SVM Classification : ', round(metrics.mean_squared_error(y_test, predictions), 5))
    print('The achieved accuracy using SVM is  ' + str(round(accuracy, 3)))
    timer.toc()
    print('SVM Classifier Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')


# ----------------------------------------------------------------------------------------------------------------------------
def Kmean_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train_PCA)
    predict = kmeans.predict(X_test1_PCA)
    accuracy = np.mean(predict == y_test) * 100
    print('Mean Square Error For K-mean Classification : ', round(metrics.mean_squared_error(y_test, predict), 5))
    print('The achieved accuracy using K-mean is  ' + str(round(accuracy, 3)))
    timer.toc()
    print('AdaBoost Classifier Time : ' +  str(round(timer.elapsed / 60, 5)) + ' Minutes')
#--------------------------------------------------------------------------------------------------------------------------
def KNN_Model_KTrials_PCA(X_train, X_test, y_train, y_test , K=40):
    timer = TicToc('timer')
    timer.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    scaler = StandardScaler()
    scaler.fit(X_train_PCA)
    X_train = scaler.transform(X_train_PCA)
    X_test = scaler.transform(X_test1_PCA)
    error = []
    accuracy=[]
    # Calculating error for K values between 1 and K
    for i in range(1, K):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        accuracy.append(np.mean(pred_i==y_test)*100)
    timer.toc()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), accuracy, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accurcy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Acc')
    plt.show()
    print('MSE Scores')
    print(error)
    print('Accurcy scores')
    print(accuracy)
    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')