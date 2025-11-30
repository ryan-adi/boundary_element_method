from common_modules import np, pd

def export_csv(csv_path, data:dict):
    pd_data = pd.DataFrame.from_dict(data)
    pd_data.to_csv(csv_path, index=False)
