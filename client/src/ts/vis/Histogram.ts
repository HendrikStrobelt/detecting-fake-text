import { VComponent } from "./VisComponent";
import { D3Sel } from "../etc/Util";
import { SimpleEventHandler } from "../etc/SimpleEventHandler";
import * as d3 from "d3";
import { schemeDark2 } from "d3";

export type HistogramData = {
    data: number[],
    label?: string,
    no_bins?: number,
    extent?: number[]
}


export class Histogram extends VComponent<HistogramData> {
    protected _current = {};
    protected css_name = 'HistogramX';
    protected options = {
        width: 200,
        height: 150,
        margin_top: 10,
        numberFormat: d3.format('.3')
    };
    highlightLabel: d3.Selection<SVGTextElement, any, any, any>;

    constructor(d3Parent: D3Sel, eventHandler?: SimpleEventHandler, options: {} = {}) {
        super(d3Parent, eventHandler);
        this.superInitSVG(options, ['bg', 'main', 'box', 'fg']);
        this._init();
    }

    protected _init() {
        const op = this.options;

        this.parent.attrs({
            width: op.width,
            height: op.height
        });

        this.layers.bg.append('g')
            .attr('class', 'y-axis')
            .attr('transform', `translate(${op.width - 33},0)`)

        this.layers.bg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${op.height - 21})`)

        this.highlightLabel = this.layers.fg.append('text')
            .attr('class', 'highlightLabel sizeLabel')

    }

    protected _render(rD: HistogramData): void {
        const op = this.options;

        const values = rD.data.map(d => +d).sort((a, b) => a - b);
        const extent = rD.extent || d3.extent(values);
        let valueScale = d3.scaleLinear().domain(extent).range([5, op.width - 35]);
        const idealNoBins = rD.no_bins || Math.min(d3.thresholdFreedmanDiaconis(values, extent[0], extent[1]), 20);
        valueScale = valueScale.nice(idealNoBins);

        const thresholds = valueScale.ticks(idealNoBins);
        if (thresholds[thresholds.length - 1] == extent[1]) thresholds.pop();

        const histo = d3.histogram()
            .domain(<[number, number]>valueScale.domain())
            .thresholds(thresholds)(values);
        console.log(histo, '-- histo');


        const countScale = d3.scaleLinear().domain([0, d3.max(histo, h => h.length)])
            .nice().range([op.height - 35, op.margin_top]);

        const adjustWidth = (bandH: number) => (bandH > 5) ? (bandH - 1) : (0.9 * bandH);

        const bars = this.layers.main.selectAll('.bar').data(histo)
            .join('rect')
            .attr('class', 'bar')
            .attrs({
                x: d => valueScale(d.x0),
                y: d => countScale(d.length),
                width: d => adjustWidth(valueScale(d.x1) - valueScale(d.x0)),
                height: d => op.height - 35 - countScale(d.length),
            })
            .on('mouseenter', d => {
                const x = valueScale(d.x0) + .5 * adjustWidth(valueScale(d.x1) - valueScale(d.x0));
                const y = countScale(d.length) - 2;

                this.highlightLabel
                    .attr('transform', `translate(${x},${y})`)
                    .style('visibility', null)
                    .text(() => d.length)
            })
            .on('mouseleave', () => this.highlightLabel.style('visibility', 'hidden'))


        this.layers.bg.select('.y-axis').call(d3.axisRight(countScale).tickFormat(op.numberFormat));
        this.layers.bg.select('.x-axis').call(d3.axisBottom(valueScale).tickFormat(op.numberFormat).ticks(thresholds.length));


        const median_d = d3.quantile(values, .5);
        const quantiles_d = [0.25, .75].map(p => d3.quantile(values, p));


        this.layers.box.selectAll('.quantiles').data([quantiles_d.map(d => valueScale(d))])
            .join('rect')
            .attr('class', 'quantiles')
            .attrs({
                x: d => d[0],
                width: d => d[1] - d[0],
                y: op.height - 33,
                height: 10
            });


        const median = this.layers.box.selectAll('.median').data([median_d])
            .join('g')
            .attr('class', 'median')
            .attr('transform', d => `translate(${valueScale(d)},${op.height - 28})`)
            .html(d => '<line x1="-3" y1="0" x2="3" y2="0"/>' +
                '<line x1="0" y1="-3" x2="0" y2="3"/>' + `<text x="6">${op.numberFormat(d)}</text>`);

    }

    protected _wrangle(data) {
        return data;
    }

}