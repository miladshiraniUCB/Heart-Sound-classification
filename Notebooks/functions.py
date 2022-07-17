def print_results(model, X_train, y_train,X_test, y_test):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
    
    ax1 = axes[0]
    plot_confusion_matrix(model, X_test, y_test, 
                          display_labels=le.classes_,
                          cmap=plt.cm.Blues, ax = ax1)
    
    ax1.set_title("Confusion Matrix for Test Set")
    
    ax2 = axes[1]
    plot_confusion_matrix(model, X_train, y_train, 
                          display_labels=le.classes_,
                          cmap=plt.cm.Blues, ax = ax2)
    
    
    ax2.set_title("Confusion Matrix for Train Set")
    
    
    
    ### Presenting Classification Report as a DataFrame
    
    train_class = classification_report(y_train, model.predict(X_train), output_dict = True)
    test_class  = classification_report(y_test, model.predict(X_test), output_dict = True)
    
    train_df = pd.DataFrame(train_class)
    test_df  = pd.DataFrame(test_class)
    
    train_df["data"] = "TRAIN"
    test_df["data"] = "TEST"
    

    report = pd.concat([test_df, train_df], axis = 0)
    report.rename(columns = {"1": f"{list(le.inverse_transform([1]))[0]}",
                             "0": f"{list(le.inverse_transform([0]))[0]}"}, inplace = True)
    report["index"] = list(report.index)

    report.set_index(["data", "index"], inplace = True)
    
    for item in list(report.columns):
        report[item] = report[item].apply(lambda x: np.round(x,2))
        
        
    score = {}
    
    score["test_recall"]  = report["abnormal"]["TEST"]["recall"]
    score["train_recall"] = report["abnormal"]["TRAIN"]["recall"]
    
    score["test_precision"]  = report["abnormal"]["TEST"]["precision"]
    score["train_precision"] = report["abnormal"]["TRAIN"]["precision"]
    
    score["test_f1-score"]  = report["abnormal"]["TEST"]["f1-score"]
    score["train_f1-score"] = report["abnormal"]["TRAIN"]["f1-score"]
    
    score["test_accuracy"]  = report["accuracy"]["TEST"]["recall"]
    score["train_accuracy"] = report["accuracy"]["TRAIN"]["recall"]
    
    return report, score