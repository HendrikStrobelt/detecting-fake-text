/*
Attn API and Types
 */

import * as d3 from "d3";
import URLHandler from "../etc/URLHandler";

export type AnalyzedText = {
    bpe_strings: string[],
    real_topk: [number, number][],
    pred_topk: [string, number][][]
}

export type AnalyzeResponse = {
    request: { project: string, text: string },
    result: AnalyzedText
}

export class GLTR_API {

    constructor(private baseURL: string = null) {
        if (this.baseURL == null) {
            this.baseURL = URLHandler.basicURL();
        }
    }


    public all_projects(): Promise<{ [key: string]: string }> {
        return d3.json(this.baseURL + '/api/all_projects')
    }


    public analyze(project: string, text: string, bitmask: number[] = null): Promise<AnalyzeResponse> {
        const payload = {
            project, text
        }
        if (bitmask) {
            payload['bitmask'] = bitmask;
        }

        return d3.json(this.baseURL + '/api/analyze', {
            method: "POST",
            body: JSON.stringify(payload),
            headers: {
                "Content-type": "application/json; charset=UTF-8"
            }
        });

    }


}

