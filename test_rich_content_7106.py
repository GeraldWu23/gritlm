import codecs
import json
import os
from pathlib import Path

import numpy as np
import requests
from retry import retry

import requests
import torch

from get_table_merger import CombineCrossPageCellTableMerger
from gritlm import GritLM
from gritlm.gritlm import sort_simi
from pdf2txt_decoder import Pdf2TxtDecoder

script_path = Path(__file__).resolve()
GRITLM_ROOT = script_path.parent
TEST_ELLM_CHAT_API_URL = 'http://imers-llm2.datagrand.cn:7106/ellm_embedding_encode'


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

@retry(exceptions=Exception, tries=5, delay=1, backoff=2, max_delay=10, logger=None)
def completion(api_provider, model, temperature, caller_request_id=None, gpt_messages=None, text_messages=None, instruction=''):
    if api_provider == 'ellm':
        assert text_messages is not None, 'text_messages should not be None while using ellm api'
        url = TEST_ELLM_CHAT_API_URL
        payload = {
            "texts": text_messages,
            "request_id": caller_request_id,
            "direct_output": True,
            "data": {
                "instruction": instruction,
                "max_length": 512

            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result_content = json.loads(response.text)
        else:
            raise RuntimeError(f'{response.status_code} {response.reason}')
    else:
        raise RuntimeError('api_provider: {} not supported'.format(api_provider))
    # print('result_content: {}'.format(result_content))
    return result_content


if __name__ == "__main__":
    instruction = "给定一个字段类型名称，返回符合该字段名称相对应的内容的段落或表格："

    info_list = [
        'http://100.100.22.51:20030/upload/mark/20230208/347975a0-a755-11ed-8c8c-02420a031656_ocr_info.json'  # 采购合同
        # "http://100.100.22.51:20030/upload/mark/20220428/3da3ed2e-c6a8-11ec-b511-02420a011e8f_ocr_info.json"  # 私募基金合同
    ]

    max_length = 512

    rich_content_file_path = download_from_info(info_list[0])
    rich_content = Pdf2TxtDecoder(load_json(rich_content_file_path), logger=None, merge_table=True,
                                  table_merger=CombineCrossPageCellTableMerger())
    element_list = rich_content.document.document_element_list
    text_list = [ele.text[:max_length] for ele in element_list]

    queries = [
        "违约责任",
        "争议解决",
        "产品单价，通常包含几个产品名称，有几个相应的数字表示单价，通常在一个表格内，包含很多中括号",
        "产品总价，通常包含几个产品名称，有几个相应的数字表示单价，通常在一个表格内，包含很多中括号",
        "合同标题，通常是什么什么合同",
        "货物运输方式",
        "运输费用承担",
        "合同份数"
        # "基金托管人信息",
        # "认购份额的计算方式",
        # "申购限制",
        # "赎回限制",
        # "投资限制",
        # "业绩报酬",
    ]

    messages = {
        'text_messages': queries,
        'instruction': "给定一个字段类型名称，返回符合该字段名称相对应的内容的段落或表格："
    }
    query_result = completion(api_provider='ellm',
                              caller_request_id='qwer',
                              model=None,
                              temperature=0.01,
                              **messages)

    messages = {
        'text_messages': text_list
    }
    text_result = completion(api_provider='ellm',
                             caller_request_id='qwer',
                             model=None,
                             temperature=0.01,
                             **messages)

    cos_sim = torch.mm(torch.from_numpy(np.array(query_result)), torch.from_numpy(np.array(text_result)).transpose(0, 1))
    # print(cos_sim.tolist())

    result_list = []
    for i in range(len(queries)):
        # print(queries[i])
        result = [text_list[j] for j in np.argsort(cos_sim[i]).tolist()[::-1]][:10]
        # print(result)
        result_list.append(result)
        _ = 100
    _ = 100



