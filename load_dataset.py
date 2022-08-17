from fnmatch import fnmatch

import datasets
from datasets import Features, concatenate_datasets, interleave_datasets, load_dataset
from datasets.filesystems import HfFileSystem
from huggingface_hub import dataset_info


data_files_with_entities = [
    "c4-en-html_cc-main-2019-18_pq00-000.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-001.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-002.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-003.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-004.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-005.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-006.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-007.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-008.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-009.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-010.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-011.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-012.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-013.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-014.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-015.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-016.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-017.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-018.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-019.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-020.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-021.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-022.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-023.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-024.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-025.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-026.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-027.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-028.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-029.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-030.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-031.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-032.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-033.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-034.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-035.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-036.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-037.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-038.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-039.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-040.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-041.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-042.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-043.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-044.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-045.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-046.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-047.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-048.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-049.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-050.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-051.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-052.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-053.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-054.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-055.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-056.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-057.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-058.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-059.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-060.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-061.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-062.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-063.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-064.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-065.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-066.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-067.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-068.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-069.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-070.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-071.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-072.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-073.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-074.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-075.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-076.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-077.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-078.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-079.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-080.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-081.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-082.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-083.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-084.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-085.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-086.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-087.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-088.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-089.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-090.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-091.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-092.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-093.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-094.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-095.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-096.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-097.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-098.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-099.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-100.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-101.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-102.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-103.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-104.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-105.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-106.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-107.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-108.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-109.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-110.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-111.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-112.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-113.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-114.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-115.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-116.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-117.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-118.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-119.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-120.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-121.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-122.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-123.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-124.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-125.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-126.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-127.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-128.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-129.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-130.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-131.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-132.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-133.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-134.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-135.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-136.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-137.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-138.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-139.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-140.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-141.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-142.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-143.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-144.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-145.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-146.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-147.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-148.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-149.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-150.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-151.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-152.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-153.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-154.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-155.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-156.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-157.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-158.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-159.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-160.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-161.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-162.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-163.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-164.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-165.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-166.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-167.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-168.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-169.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-170.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-171.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-172.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-173.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-174.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-175.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-176.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-177.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-178.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-179.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-180.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-181.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-182.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-183.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-184.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-185.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-186.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-187.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-188.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-189.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-190.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-191.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-192.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-193.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-194.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-195.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-196.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-197.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-198.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-199.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-200.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-201.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-202.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-203.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-204.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-205.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-206.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-207.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-208.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-209.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-210.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-211.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-212.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-213.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-214.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-215.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-216.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-217.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-218.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-219.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-220.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-221.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-222.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-223.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-224.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-225.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-226.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-227.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-228.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-229.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-230.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-231.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-232.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-233.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-234.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-235.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-236.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-237.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-238.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-239.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-240.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-241.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-242.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-243.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-244.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-000.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-001.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-002.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-003.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-004.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-005.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-006.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-007.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-008.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-009.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-010.jsonl",
    "c4-en-html_cc-main-2019-18_pq01-011.jsonl",
]

features = {
    "HtmlPreprocessor_error": {"_type": "Value", "dtype": "int64", "id": None},
    "HtmlPreprocessor_error_comment": {"_type": "Value", "dtype": "string", "id": None},
    "c4_shard": {"_type": "Value", "dtype": "int64", "id": None},
    "c4_timestamp": {"_type": "Value", "dtype": "string", "id": None},
    "html": {"_type": "Value", "dtype": "string", "id": None},
    "html_footer": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_head": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_title": [{"dtype": "string", "id": None, "_type": "Value"}],
    "metadata_generation_datasource": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_generation_length_sentence": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_generation_length_text": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_html": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "html_attrs": {
                "attrs": [{"_type": "Value", "dtype": "string", "id": None}],
                "values": [{"_type": "Value", "dtype": "string", "id": None}],
            },
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "relative_end_pos": {"_type": "Value", "dtype": "int64", "id": None},
            "relative_start_pos": {"_type": "Value", "dtype": "int64", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_paragraph": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "marker": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_timestamp": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_title": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_url": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_website_desc": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "text": {"_type": "Value", "dtype": "string", "id": None},
    "url": {"_type": "Value", "dtype": "string", "id": None},
}
features_with_entities = features.copy()
features_with_entities.update(
    {
        "metadata_entity": [
            {
                "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "key": {"_type": "Value", "dtype": "string", "id": None},
                "type": {"_type": "Value", "dtype": "string", "id": None},
                "value": {"_type": "Value", "dtype": "string", "id": None},
            }
        ],
        "metadata_entity_paragraph": [
            {
                "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "key": {"_type": "Value", "dtype": "string", "id": None},
                "relative_end_pos": {"_type": "Value", "dtype": "int64", "id": None},
                "relative_start_pos": {"_type": "Value", "dtype": "int64", "id": None},
                "type": {"_type": "Value", "dtype": "string", "id": None},
                "value": {"_type": "Value", "dtype": "string", "id": None},
            }
        ],
    }
)


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        try:
            return getattr(datasets, features["_type"])(features["dtype"])
        except ValueError:
            print(features)
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


new_features = {}
final_features = convert_types(features)
final_features_with_entities = convert_types(features_with_entities)

# di = dataset_info("bs-modeling-metadata/c4-en-html-with-metadata")
# fs = HfFileSystem(di)
# all_files = fs.ls(".")


def get_files(pattern):
    for file in all_files:
        if fnmatch(file, pattern):
            yield file


def load_dataset_by_files(files, dataset_name_or_path="bs-modeling-metadata/c4-en-html-with-metadata"):
    selected_files_entities = list(filter(lambda v: v in data_files_with_entities, files))
    selected_files_no_entities = list(filter(lambda v: v not in data_files_with_entities, files))
    datasets = []
    if selected_files_entities:
        dataset_entities = load_dataset(
            dataset_name_or_path,
            features=Features(final_features_with_entities),
            data_files=selected_files_entities,
            split="train",
            use_auth_token=True,
            ignore_verifications=True
        )
        print("dataset_entities", dataset_entities)
        datasets.append((dataset_entities, len(selected_files_entities)))

    if selected_files_no_entities:
        dataset_no_entities = load_dataset(
            dataset_name_or_path,
            features=Features(final_features),
            data_files=selected_files_no_entities,
            split="train",
            use_auth_token=True,
            ignore_verifications=True
        )
        print("dataset_no_entities", dataset_no_entities)
        datasets.append((dataset_no_entities, len(selected_files_no_entities)))
    dataset = concatenate_datasets([d for d, _ in datasets])
    return dataset
