import codecs
import json
import os
from pathlib import Path

import requests
import torch

from get_table_merger import CombineCrossPageCellTableMerger
from gritlm import GritLM
from gritlm.gritlm import sort_simi
from pdf2txt_decoder import Pdf2TxtDecoder

script_path = Path(__file__).resolve()
GRITLM_ROOT = script_path.parent


def load_json(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    return json_dict


def download_from_info(info):
    file_dir = os.path.join(GRITLM_ROOT, 'test_data')
    file_name = info.split('/')[-1].rstrip('info.json') + 'all.json'
    file_url = info.rstrip('info.json') + 'all.json'
    path = os.path.join(file_dir, file_name)

    if not os.path.exists(path):
        # 发起 HTTP GET 请求
        # 确保响应成功
        try:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                # 确保目录存在
                os.makedirs(file_dir, exist_ok=True)

                with open(path, 'wb') as file:
                    # 分块写入文件，防止内存过载
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉保持活动的新块
                            file.write(chunk)
            else:
                print(f'下载失败，状态码: {response.status_code}')
        except Exception as e:
            print(f'\n下载文件失败: {e}')
            return path

    return path


if __name__ == "__main__":
    info_list = [
        # 'http://100.100.22.51:20030/upload/mark/20230208/347975a0-a755-11ed-8c8c-02420a031656_ocr_info.json'  # 采购合同
        "http://100.100.22.51:20030/upload/mark/20220428/3da3ed2e-c6a8-11ec-b511-02420a011e8f_ocr_info.json"  # 私募基金合同
    ]

    max_length = 512

    rich_content_file_path = download_from_info(info_list[0])
    rich_content = Pdf2TxtDecoder(load_json(rich_content_file_path), logger=None, merge_table=True,
                                  table_merger=CombineCrossPageCellTableMerger())
    element_list = rich_content.document.document_element_list
    text_list = [ele.text[:max_length] for ele in element_list]

    instruction = "给定一个字段类型名称，返回符合该字段名称相对应的内容的段落或表格："
    queries = [
        # "违约责任",
        # "争议解决",
        # "产品单价，通常包含几个产品名称，有几个相应的数字表示单价，通常在一个表格内，包含很多中括号",
        # "产品总价，通常包含几个产品名称，有几个相应的数字表示单价，通常在一个表格内，包含很多中括号",
        # "合同标题，通常包含几个产品名称，有几个相应的数字表示单价，通常在一个表格内，包含很多中括号",
        # "货物运输方式",
        # "运输费用承担",
        "基金托管人信息",
        "认购份额的计算方式",
        "申购限制",
        "赎回限制",
        "投资限制",
        "业绩报酬",
    ]

    model = GritLM(
        model_name_or_path = '/data/models/GritLM-7B',
        # model_name_or_path = '/data/models/GritLM-8x7B',
        # model_name_or_path='/data/models/Yi-34B',
        # mode = 'embedding', # One of ['unified', 'embedding', 'generative']
        pooling_method='weightedmean',  # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
        normalized=True,
        projection=None,
        is_inference=True,
        embed_eos="",
        attn='bbcc',

        # from base ellm server
        mode='unified',
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map='auto'
    )

    result_list = sort_simi(model=model, queries=queries, documents=text_list, instruction=instruction)
    print([(q, r) for q, r in zip(queries, result_list)])

