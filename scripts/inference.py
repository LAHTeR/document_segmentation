import argparse
from pathlib import Path

import torch

from document_segmentation.model.dataset import PageDataset
from document_segmentation.model.device_module import DeviceModule
from document_segmentation.model.page_sequence_tagger import PageSequenceTagger
from document_segmentation.pagexml.datamodel.label import Label
from document_segmentation.pagexml.datamodel.page import Page
from document_segmentation.pagexml.inventory import InventoryReader

if __name__ == "__main__":
    ########################################################################################
    # PARSE ARGUMENTS
    ########################################################################################
    arg_parser = argparse.ArgumentParser(
        description="Apply a model to one or more inventories."
    )
    arg_parser.add_argument(
        "--inventory", nargs="+", type=str, help="The inventory number(s)."
    )
    arg_parser.add_argument(
        "--model-file",
        type=argparse.FileType("rb"),
        default="model.pt",
        help="The model file (default: 'model.pt')",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=".",
        help="Output directory. Defaults to current directory.",
    )

    arg_parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("xt"),
        default="-",
        help="Output file. Defaults to stdout.",
    )
    arg_parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")

    arg_parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=DeviceModule.get_device(),
        help="The device to use for training. Defaults to 'cpu'.",
    )

    args = arg_parser.parse_args()

    device = torch.device(args.device)
    model: PageSequenceTagger = torch.load(args.model_file, map_location=device)
    model.to_device(device)

    model.train(False)

    for inv_nr in args.inventory:
        inventory = InventoryReader(inv_nr)
        dataset: PageDataset = PageDataset(
            [
                Page.from_pagexml(Label.BEGIN, scan_nr=i, pagexml=page_xml)
                for i, page_xml in enumerate(inventory.all_pagexmls(), start=1)
            ]
        ).remove_short_regions()

        for batch in dataset.batches(args.batch_size):
            labels: torch.Tensor = model(batch)
            output = torch.argmax(labels, dim=1)
            for page, label in zip(batch, output):
                _label = Label(label.item())
                # page.label = _label
                print(
                    page.doc_id,
                    _label.name,
                    page.text(delimiter="; ")[:50],
                    file=args.output,
                )

        # documents: Iterable[Document] = Document.from_pages(dataset.pages, inv_nr)
        # args.output_dir.mkdir(parents=True, exist_ok=True)
        # for doc in documents:
        #     _path: Path = (args.output_dir / f"{inv_nr}_{doc.id}").with_suffix(".json")
        #     with _path.open("xt") as f:
        #         f.write(doc.model_dump_json())
