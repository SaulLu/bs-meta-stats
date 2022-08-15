def is_not_empty_list(values):
    return {"not_none": list(filter(lambda v: v!=[], values))}

def is_not_none(values):
    return {"not_none": list(filter(lambda v: v is not None, values))}

def get_num_timestamp(ds):
    column_names = list(ds.features.keys())
    sub_ds = ds.map(is_not_empty_list, batched=True, remove_columns=column_names, input_columns=["metadata_timestamp"])
    return len(sub_ds)

def get_num_website_desc(ds):
    column_names = list(ds.features.keys())
    sub_ds = ds.map(is_not_empty_list, batched=True, remove_columns=column_names, input_columns=["metadata_website_desc"])
    return len(sub_ds)