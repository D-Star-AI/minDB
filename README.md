# minDB -  an extremely memory-efficient vector database
Existing open source vector databases are built on HNSW indexes that must be held entirely in memory to be used. This uses an extremely large amount of memory, which severely limits the sizes of vector DBs that can be used locally, and creates very high costs for cloud deployments.

It’s possible to build a vector database with extremely low memory requirements that still has high recall and low latency. The key is to use a highly compressed search index, combined with reranking from disk, as demonstrated in the [Zoom](https://arxiv.org/abs/1809.04067) paper. This project implements the core technique introduced in that paper. We also implement a novel adaptation of Faiss's two-level k-means clustering algorithm that only requires a small subset of vectors to be held in memory at an given point.

With minDB, you can index and query 100M 768d vectors with peak memory usage of around 3GB. With an in-memory vector DB, you would need ~340GB of RAM. This means you could easily index and query all of Wikipedia on an average Macbook.

**Note:** This is currently in a beta phase, and has not been fully tested in a production environment. It is possible there are bugs or edge cases that have not been tested, or additional limitations that are not listed here. There may also be breaking changes in the future.

## Architecture overview
minDB uses a two-step process to perform approximate nearest neighbors search. First, a highly compressed Faiss index is searched to find the `preliminary_top_k` (set to 500 by default) results. Then the full uncompressed vectors for these results are retrieved from a key-value store on disk, and a k-nearest neighbors search is performed on these vectors to arrive at the `final_top_k` results.

## Basic usage guide

Clone the repo and run `pip install -r requirements.txt` to install all of the necessary packages.

For a quickstart guide, check out our getting started example [here](https://github.com/SuperpoweredAI/minDB/blob/main/examples/getting_started.ipynb).

By default, all minDB databases are saved to the ~/.spdb directory. This directory is created automatically if it doesn’t exist when you initialize an minDB object. You can override this path by specifying a save_path when you create your minDB object.

## Adding and removing items
To add vectors to your database, call the `/db/{db_name}/add` endpoint, or use the `db.add()` method. This takes a list of `(vector, metadata)` tuples, where each vector is itself a list, and each metadata item is a dictionary with keys of your choosing.

To remove items, you must pass in a list of ids corresponding to the vectors that you want removed if using FastAPI, or an array of ids when directly using minDB. 

## How the index is trained
The search index will get automatically trained once the number of vectors exceeds 25,000 (this parameter is configurable in the params.py file). Future training operations occur once the index coverage ratio is below 0.5 (also configurable). You can also train a database at any point by calling the `/db/{db_name}/train` function, or the `db.train()` method if you're not running FastAPI. If there are fewer than 5,000 vectors in a database, the training operation will be skipped and a flat index will still be used. 

We don't recommend setting the number of vectors before training much higher than 25,000 due to increased memory usage and query latency. Untrained indexes use a flat Faiss index, which means the full uncompressed vectors are held in memory. Searches over a flat index are done via a brute force method, which gets substantially slower as the number of vectors increases.

## Metadata
You can add metadata to each vector by including a metadata dictionary. You can include whatever metadata fields you want, but the keys and values should all be serializable.

Metadata filtering is the next major feature that will be added. This will allow you to use SQL-like statements to control which items get searched over.

## FastAPI server deployment
To deploy your database as a server with a REST API, you can make use of the `fastapi.py` file. To start the server, open up a terminal and run the following command:
`uvicorn api.fastapi:app --host 0.0.0.0 --port 8000`.
Please note, you must be in the main minDB directory to run this command.

For more detail, you can check out our FastAPI tutorial [here](https://github.com/D-Star-AI/minDB/tree/main/examples/fastapi_example.ipynb).
You can also learn more about FastAPI [here](https://fastapi.tiangolo.com).

## Limitations
- One of the main dependencies, Faiss, doesn't play nice with Apple M1/M2 chips. You may be able to get it to work by building it from source, but we haven't successfully done so yet.
- We haven't tested it on datasets larger than 35M vectors yet. It should still work well up to 100-200M vectors, but beyond that performance may start to deteriorate.

## Additional documentation
- [Tunable parameters](https://github.com/D-Star-AI/minDB/wiki/Tunable-parameters)
- [Contributing](https://github.com/D-Star-AI/minDB/wiki/Contributing)
- [Examples](https://github.com/D-Star-AI/minDB/tree/main/examples)