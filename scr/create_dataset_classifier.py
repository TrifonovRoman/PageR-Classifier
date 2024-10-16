import json
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N
from pager.page_model.sub_models.dtype import Style, StyleWord, ImageSegment
import os
import argparse

page_model = PageModel([
    PageModelUnit("words_and_styles", sub_model=WordsAndStylesModel(), extractors=[], converters={}),
    PageModelUnit("graph", sub_model=SpGraph4NModel(), extractors=[],
                  converters={"words_and_styles": WordsAndStylesToSpGraph4N()}),
])


def is_one_block(word1, word2, blocks):
    for block in blocks:
        if block.is_intersection(word1) and block.is_intersection(word2):
            return 1
    return 0


def get_subgraphs_from_blocks(blocks, graph):
    subgraphs = []

    words = [w.segment for w in page_model.page_units[0].sub_model.words]

    for block in blocks:
        subgraph = {
            "A": [[], []],
            "nodes_feature": [],
            "edges_feature": [],
            "label": block.get_info('label')
        }

        block_nodes = []
        for i, word in enumerate(words):
            if block.is_intersection(word):
                block_nodes.append(i)
                subgraph["nodes_feature"].append(graph["nodes_feature"][i])

        for i, (n1, n2) in enumerate(zip(graph["A"][0], graph["A"][1])):
            if n1 in block_nodes and n2 in block_nodes:
                subgraph["A"][0].append(block_nodes.index(n1))
                subgraph["A"][1].append(block_nodes.index(n2))
                subgraph["edges_feature"].append(graph["edges_feature"][i])

        subgraphs.append(subgraph)

    return subgraphs

def get_block(bl):
    b = ImageSegment(dict_p_size=bl)
    b.add_info('label', bl['label'])
    return b

def get_graphs_from_file(file_name):
    with open(file_name, "r") as f:
        info_img = json.load(f)

    publaynet_rez = info_img["blocks"]
    pager_rez = info_img["additional_info"]

    page_model.from_dict(pager_rez)
    page_model.extract()
    graph = page_model.to_dict()

    blocks = [get_block(bl) for bl in publaynet_rez]

    subgraphs = get_subgraphs_from_blocks(blocks, graph)

    return subgraphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create classification dataset from PubLayNet and PageR')
    parser.add_argument('--path_dir_jsons', type=str, nargs='?', required=True,
                        help='Path to the directory with json dataset')
    parser.add_argument('--path_rez_json', type=str, nargs='?', required=True,
                        help='Path to the resulting json file')

    args = parser.parse_args()
    args.path_dir_jsons,
    args.path_rez_json

    dataset = {
        "dataset": []
    }

    files = os.listdir(args.path_dir_jsons)
    N = len(files)

    for i, json_file in enumerate(files):
        subgraphs = []
        try:
            subgraphs = get_graphs_from_file(os.path.join(args.path_dir_jsons, json_file))
        except:
            print(json_file)
        for subgraph in subgraphs:
            dataset["dataset"].append(subgraph)

        print(f"{(i + 1) / N * 100:.2f} %" + 20 * " ", end='\r')

    with open(args.path_rez_json, "w") as f:
        json.dump(dataset, f)
