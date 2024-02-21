import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path

valid_fields = [
        'station',          ## ASOS station contributing measurement
        'valid',            ## Data acquisition time
        'tmpf',             ## Temperature (F)
        'tmpc',             ## Temperature (C)
        'dwpf',             ## Dewpoint (F)
        'dwpc',             ## Dewpoint (C)
        'relh',             ## Relative humidity (%)
        'feel',             ## Feels like temperature (F)
        'sknt',             ## Wind speed (kt)
        'mslp',             ## Mean sea level pressure (hPa)
        'p01m',             ## Precipitation (mm)
        'p01i',             ## Precipitation (in)
        'gust',             ## Gust speed (kt) (!! has null values !!)
        'skyc1',            ##
        'skyl1',            ## Sky conditions at level 1
        'romps_LCL_m',      ## LCL from analytic calculation
        'lcl_estimate'      ## LCL from empirical estimate
        ]

def parse_csv(csv_path:Path, fields:list=None, replace_val=np.nan):
    """
    Parse a monthly ASOS report for a user specified number/ordering of fields,
    converting to datetime or float value where appropriate, and replacing
    missing float values with the requested number.

    :@param csv_path: valid path to ASOS style csv
    :@param fields: Optionally specify a subset of the fields in parsing order.
    :@param replace_val: Value to replace null entries in float fields.
    """
    str_fields = ["station", "skyc1", "skyl1",]
    time_fields = ["valid"]
    all_lines = csv_path.open("r").readlines()
    labels = all_lines.pop(0).strip().split(",")
    all_cols = list(zip(*map(lambda l:l.strip().split(","),all_lines)))
    if fields is None:
        fields = labels
    else:
        assert all(f in labels for f in fields)

    data = []
    for f in fields:
        idx = labels.index(f)
        if labels[idx] in time_fields:
            data.append(tuple(map(
                lambda t:datetime.strptime(t,"%Y-%m-%d %H:%M"),
                all_cols[idx]
                )))
        elif labels[idx] not in str_fields:
            data.append(tuple(map(float,
                ([v,replace_val][v==""] for v in all_cols[idx])
                )))
        else:
            data.append(tuple(all_cols[idx]))
    return fields,data

def get_norm_coeffs(data):
    """
    Calculate the mean and standard deviation of each feature

    :@param data: list of uniform-size data arrays corresponding to each field
    :@return: tuple[np.array] like (means, stdevs)
    """
    data = np.stack(data)
    return (np.average(data, axis=-1), np.std(data,axis=-1))

def preprocess(asos_csv_path:Path, input_feats:list, output_feats:list,
               pkl_path:Path=None, normalize=False):
    """
    Preprocess an ASOS monthly csv file to
    """
    ## Always load the station name and acquisition time
    default_fields = ["station", "valid"]
    all_fields = list(set(input_feats+output_feats+default_fields))
    assert all(f in valid_fields for f in all_fields)
    labels,data = parse_csv(
            csv_path=asos_csv_path,
            fields=all_fields,
            replace_val=0, ## gust should be the only NaN field.
            )
    ## Parse out the station name and acquisition time
    stations = data.pop(labels.index("station"))
    labels.remove("station")
    times = data.pop(labels.index("valid"))
    labels.remove("valid")
    means,stdevs = get_norm_coeffs(data)
    data = np.stack(data,axis=-1) ## shape (samples,features)
    ## normalize the data if requested
    if normalize:
        data = (data-means)/stdevs
    ## split into input and output datasets
    X,x_means,x_stdevs = map(np.asarray, zip(*[(
        data[...,labels.index(f)],
        means[...,labels.index(f)],
        stdevs[...,labels.index(f)]
        ) for f in input_feats]))
    Y,y_means,y_stdevs = map(np.asarray, zip(*[(
        data[...,labels.index(f)],
        means[...,labels.index(f)],
        stdevs[...,labels.index(f)]
        ) for f in output_feats]))
    dataset = {
            "X":X.T,
            "x_labels":input_feats,
            "x_means":x_means,
            "x_stdevs":x_stdevs,

            "Y":Y.T,
            "y_labels":output_feats,
            "y_means":y_means,
            "y_stdevs":y_stdevs,

            "stations":stations,
            "times":times,
            "normalized":normalize,
            }
    if not pkl_path is None:
        pkl.dump(dataset, pkl_path.open("wb"))
    return dataset

if __name__=="__main__":
    data_dir = Path("data")
    csv_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
    pkl_path = data_dir.joinpath("202306_asos_al.pkl")

    data_dict = preprocess(
            asos_csv_path=csv_path,
            input_feats=["tmpc","dwpc","relh","sknt","mslp","p01m","gust"],
            output_feats=["romps_LCL_m","lcl_estimate"],
            pkl_path=pkl_path,
            normalize=True,
            )
    print(f"Dict keys: {tuple(data_dict.keys())}")
    print(f"Inputs:    {data_dict['X'].shape} {data_dict['x_labels']}")
    print(f"Predicted: {data_dict['Y'].shape}  {data_dict['y_labels']}")
