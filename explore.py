#%%
from collections import Counter, defaultdict
from load_dataset import load_dataset_by_files
#%%
files = [
    "c4-en-html_cc-main-2019-18_pq00-000.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-001.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-002.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-003.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-004.jsonl",
]
# %%
dataset_path = '/home/lucile_huggingface_co/data/c4-en-html-with-metadata'
# %%
ds = load_dataset_by_files(files=files, dataset_name_or_path=dataset_path)
# %%
ds
# %%
ds[0]["metadata_entity_paragraph"]

# %%
def is_not_empty_list(values):
    not_none = list(filter(lambda v: v!=[], values))
    return {"not_none": not_none, "num": [len(item) for item in not_none]}
# %%
def is_not_none(values):
    return {"not_none": list(filter(lambda v: v is not None, values))}
#%%
column_names = list(ds.features.keys())
#%%
sub_ds = ds.map(is_not_empty_list, batched=True, remove_columns=column_names, input_columns=["metadata_timestamp"], load_from_cache_file=False)
# %%
len(sub_ds)
# %%
sub_ds["not_none"]
# %%

sub_ds["num"]
# %%
sub_ds
# %%
# HTML_TAGS = Counter()
# def select_col(html_metadata):
#     for metadata_ind in html_metadata:
#         for metadata in metadata_ind:
#             for html_tag in metadata["value"]:
#                 HTML_TAGS[html_tag] += 1
#     return {"html_metadata":[]}

# sub_ds = ds.map(
#     select_col, 
#     batched=True, 
#     remove_columns=column_names, 
#     input_columns=["metadata_html"], 
#     load_from_cache_file=False
# )
# %%

def extract_html_tags(html_metadata):
    return {"html_tags":[metadata["value"] for metadata_ind in html_metadata for metadata in metadata_ind]}

sub_ds = ds.map(
    extract_html_tags, 
    batched=True, 
    num_proc=8,
    remove_columns=column_names, 
    input_columns=["metadata_html"], 
    # load_from_cache_file=False
)

HTML_TAGS = Counter()
def count_html_tags(html_tags):
    for html_tag in html_tags:
        HTML_TAGS[html_tag] += 1
    return {"html_metadata":[]}

_sub_ds = sub_ds.map(
    count_html_tags, 
    batched=True, 
    batch_size=100_000,
    remove_columns=["html_tags"], 
    input_columns=["html_tags"], 
    load_from_cache_file=False
)
print(HTML_TAGS)
# %%

def html_tags_per_doc(html_metadata):
    return {"html_tags":[set(metadata["value"] for metadata in metadata_ind) for metadata_ind in html_metadata]}

sub_ds = ds.map(
    html_tags_per_doc, 
    batched=True, 
    num_proc=8,
    remove_columns=column_names, 
    input_columns=["metadata_html"], 
    load_from_cache_file=False
)
# %%

HTML_TAGS_PER_DOC = Counter()
def count_html_tags_per_doc(html_tags):
    for html_tag_set in html_tags:
        for html_tag in html_tag_set:
            HTML_TAGS_PER_DOC[html_tag] += 1
    return {"html_metadata":[]}

_sub_ds = sub_ds.map(
    count_html_tags_per_doc, 
    batched=True, 
    remove_columns=["html_tags"], 
    input_columns=["html_tags"], 
    load_from_cache_file=False
)
print(HTML_TAGS_PER_DOC)
# %%
max(HTML_TAGS_PER_DOC, key=HTML_TAGS_PER_DOC.get)
# %%
max(HTML_TAGS, key=HTML_TAGS.get)
#%%
print(HTML_TAGS)
print(HTML_TAGS_PER_DOC)
# %%
ds[0]["metadata_generation_datasource"]
# %%
ds[1]["metadata_generation_datasource"]
# %%
for i in range(1000):
    if ds[i]["metadata_timestamp"] != []:
        print(i, ds[i]["metadata_timestamp"][0]["value"])
        # break
# %%
metadata_names = [name for name in ds.features.keys() if name.startswith("metadata_")] 
# %%
metadata_names
# %%
for name in metadata_names:
    print(name, type(ds[0][name]))
# %%
def is_not_empty_list(values):
    return {"not_none": list(filter(lambda v: v!=[], values))}

def get_num(ds, metadata_col_name, num_proc):
    column_names = list(ds.features.keys())
    sub_ds = ds.map(is_not_empty_list, batched=True, num_proc=num_proc, remove_columns=column_names, input_columns=[metadata_col_name])
    return len(sub_ds)
# %%

for name in metadata_names:
    print(name, get_num(ds, metadata_col_name=name, num_proc=16))
# %%
