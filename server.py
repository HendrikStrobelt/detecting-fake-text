#!/usr/bin/env python

import argparse
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# from backend.Project import Project # TODO !!
from backend import AVAILABLE_MODELS

__author__ = 'Hendrik Strobelt, Sebastian Gehrmann'

CONFIG_FILE_NAME = 'lmf.yml'
projects = {}

app = FastAPI(debug=False)

##### CLI ARGS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt-2-small')
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="127.0.0.1")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))

parser.add_argument("--no_cors", action='store_true')

args, _ = parser.parse_known_args()

if not args.no_cors:
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class Project:
    def __init__(self, LM, config):
        self.config = config
        self.lm = LM()


@app.get("/api/all_projects")
def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].config
    return res


class AnalyzeRequest(BaseModel):
    project: str
    text: str


@app.post("/api/analyze")
def analyze(analyze_request: AnalyzeRequest):
    project = analyze_request.project
    text = analyze_request.text

    res = {}
    if project in projects:
        p = projects[project]  # type: Project
        res = p.lm.check_probabilities(text, topk=20)

    return {
        "request": {'project': project, 'text': text},
        "result": res
    }


#####
# START
# #######

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        print("Model {} not found. Make sure to register it.".format(
            args.model))
        print("Loading GPT-2 instead.")
        model = AVAILABLE_MODELS['gpt-2']
    projects[args.model] = Project(model, args.model)


#########################
#  some non-logic routes
#########################

app.mount("/client", StaticFiles(directory='client/dist'), name="client_static")
# app.mount("/data", StaticFiles(directory=args.dir), name="data_static")


@app.get('/')
def redirect():
    return RedirectResponse('client/index.html')
