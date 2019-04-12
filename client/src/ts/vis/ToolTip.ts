import { D3Sel, token_cleanup } from "../etc/Util";
import { SimpleEventHandler } from "../etc/SimpleEventHandler";
import { GLTR_RenderItem } from "./GLTR_Text_Box";
import * as d3 from "d3";


export class ToolTip {
    private predictions: D3Sel;
    private myDetail: D3Sel;

    constructor(private parent: D3Sel, private eh: SimpleEventHandler) {
        this._init()
    }


    private _init() {
        this.predictions = this.parent.select('.predictions');
        this.myDetail = this.parent.select('.myDetail');

    }


    set visility(vis: boolean) {
        if (vis == true) {
            this.parent.style('opacity', 1);
        } else {
            this.parent.style('opacity', 0);
        }
    }


    updateData(ri: GLTR_RenderItem) {
        this.visility = true;

        const maxW = ri.others[0][1];
        const wScale = d3.scaleLinear().domain([0, maxW]).range([0, 60])
        const numF = d3.format('.3f')

        let [x, y] = d3.mouse(<HTMLElement>d3.select('body').node());

        const over_half = x > window.innerWidth / 2;
        x = over_half ? x - 200 : x;

        // console.log(x, y, "--- x,y");
        // console.log(this.parent.node(),"--- this.parent.node()");
        this.parent.styles({
            top: (y + 30) + 'px',
            left: x + 'px',
        })

        this.predictions.selectAll('.row').data(ri.others.slice(0, 5))
            .join('div')
            .attr('class', 'row')
            .style('display', 'table-row')
            .html(d => {
                const color = ri.token != d[0] ? "#333" : "#933";
                const bar = '<div style="display: table-cell; width:110px;padding-left:5px;">' +
                    `<div style="display:inline-block;width: ${wScale(d[1])}px;background-color:${color};height: 10px;"></div>` +
                    ` <div style="display:inline-block;color: ${color};">${numF(d[1])}</div>` + "</div>";


                const text = `<div style="display: table-cell;color: ${color}">${token_cleanup(d[0])}</div>`;
                return `${bar} ${text}`
            })

        this.myDetail.html(() => {
            const diff = ri.others[0][1] - ri.prop;
            const frac = ri.prop / (ri.others[0][1]);

            const tk = `<span style="color: #666">top_k pos:</span> <span style="color: #333">${ri.top}</span>`
            const prop = `<span style="color: #666666">prob:</span> <span style="color: #333">${numF(ri.prop)}</span>`
            const p_share = `<span style="color: #666">frac(p):</span> <span style="color: #333">${numF(frac)}</span>`


            return `${tk} ${prop} <br/>${p_share}`

        })


    }


}