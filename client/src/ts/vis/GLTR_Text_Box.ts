import {VComponent} from "./VisComponent";
import {AnalyzedText} from "../api/GLTR_API";
import {D3Sel, token_cleanup} from "../etc/Util";
import {SimpleEventHandler} from "../etc/SimpleEventHandler";
import * as d3 from "d3";


export enum GLTR_Mode {
    topk, diff_p, fract_p
}

export type GLTR_RenderItem = { top: number; prop: number; others: [string, number][]; token: string };
export type GLTR_HoverEvent = { hovered: boolean, d: GLTR_RenderItem }

export class GLTR_Text_Box extends VComponent<AnalyzedText> {
    protected _current = {
        maxValue: -1,
    };
    protected css_name = "LMF";
    protected options = {
        gltrMode: GLTR_Mode.topk,
        diffScale: d3.scalePow<string>().exponent(.3).range(["#b4e876", "#fff"]),
        fracScale: d3.scaleLinear<string>().range(["#fff", "#b4e876"]),
        topkScale: d3.scaleThreshold<number, string>().domain([10, 100, 1000]).range([
            '#ADFF80',
            '#FFEA80',
            // '#ffbe80',
            '#FF9280',
            '#E5B0FF'
        ])
    };

    static events = {
        tokenHovered: 'lmf-view-token-hovered'
    }


    constructor(parent: D3Sel, eventHandler?: SimpleEventHandler, options = {}) {
        super(parent, eventHandler);
        this.superInitHTML(options);
        this._init();

    }

    protected _init() {


    }


    protected _render(rd: AnalyzedText = this.renderData): void {
        if (!rd) return;

        const op = this.options;
        const cur = this._current;

        const tokens = rd.bpe_strings.slice(1);
        const topK = rd.real_topk;


        const td: GLTR_RenderItem[] = tokens.map((token, i) => ({
            token,
            top: topK[i][0],
            prop: topK[i][1],
            others: rd.pred_topk[i]
        }));

        const coloring = (d: GLTR_RenderItem) => {

            if (op.gltrMode === GLTR_Mode.topk) {
                return op.topkScale(d.top)
            }

            if (op.gltrMode === GLTR_Mode.diff_p) {
                const diff = d.others[0][1] - d.prop;
                console.log(diff, op.diffScale(diff), op.diffScale.domain(), "--- diff, cur.diffScale(diff), cur.diffScale.domain()");
                return op.diffScale(diff);
            }

            if (op.gltrMode === GLTR_Mode.fract_p) {
                const frac = d.prop / (d.others[0][1]);
                return op.fracScale(frac);
            }


        }

        td.forEach(d => {
            console.log("-hen--", d.token, d.token.charCodeAt(0));

        })

        this.base.selectAll('.token').data(td, (d: GLTR_RenderItem, i) => d.token + '__' + i)
            .join('div')
            .attr('class', d => `token ${d.token.startsWith('Ġ') ? 'spaceLeft' : ''} ${d.token.startsWith('Ċ') ? 'newLine' : ''}`)
            .style('background-color', coloring)
            .text(d => token_cleanup(d.token))
            .on('mousemove', d =>
                this.eventHandler.trigger(GLTR_Text_Box.events.tokenHovered, <GLTR_HoverEvent>{
                    hovered: true,
                    d
                }))
            .on('mouseleave', d =>
                this.eventHandler.trigger(GLTR_Text_Box.events.tokenHovered, <GLTR_HoverEvent>{
                    hovered: false,
                    d
                }))


    }

    protected _wrangle(data: AnalyzedText) {
        const allTop1 = data.pred_topk.map(tk => tk[0][1]);
        this._current.maxValue = d3.max(allTop1);
        this.options.diffScale.domain([0, this._current.maxValue]);

        return data;
    }

    public updateThresholdValues(ths: number[]) {
        this.options.topkScale.domain(ths);
        this._render();
    }


    public get colorStats() {
        const res: { [key: string]: number } = {};
        this.options.topkScale.range().forEach(c => res[c] = 0);
        this.data.real_topk.map(d => d[0]).forEach(x => {
            const c = this.options.topkScale(x);
            res[c] += 1;
        })

        return {
            colors: this.options.topkScale.range(),
            values: this.options.topkScale.range().map(c => res[c])
        }
    }

}