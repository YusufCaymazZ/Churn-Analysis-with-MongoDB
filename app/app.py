
def main():
    from analysis import AnalysisTools
    from churns import Churns  # where to import
    
    churn_analysis = Churns()
    if churn_analysis.to_collections.lake is not None:
        churn_analysis.clear_lake()
        print("\n\t *******Lake dropped.*******n")
    if churn_analysis.to_collections.warehouse is not None:
        churn_analysis.clear_warehouse()
        print("\n\t *******Warehouse dropped.*******n")
    if churn_analysis.to_collections.predicted_data is not None:
        churn_analysis.clear_predicted_data
        print("\n\t *******Predicted Data dropped.*******n")

    churn_analysis.send_to_lake()
    print("\t *******sent******* \n")
    analys = AnalysisTools()
    analys.analys_data()
    


if __name__ == "__main__":
    main()
   




    
    



