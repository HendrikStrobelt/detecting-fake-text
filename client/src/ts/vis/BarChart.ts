import { VComponent } from "./VisComponent" 
import { D3Sel } from "../etc/Util";
import { SimpleEventHandler } from "../etc/SimpleEventHandler";
import { tickStep } from "d3";
import * as d3 from "d3";


export type BarChartData = {
    values: number[],
    label?: string[],
    extent?: number[],
    colors?: string[]
}

export class BarChart extends VComponent<BarChartData>{
    protected options = {
        width: 200,
        height: 150,
        margin_top: 10,
        numberFormat: d3.format('.3')
    };

    protected css_name = "barchartX";
    protected _current = {};
    highlightLabel: d3.Selection<SVGTextElement, any, any, any>;

    constructor(parent: D3Sel, eventHandler?: SimpleEventHandler, options = {}) {
        super(parent, eventHandler);
        this.superInitSVG(options);
        this._init();
    }

    protected _init() {
        const op = this.options;
        this.parent.attrs({ width: op.width, height: op.height });

        this.layers.bg.append('g')
            .attr('class', 'y-axis')
            .attr('transform', `translate(${op.width - 33},0)`);

        this.highlightLabel = this.layers.fg.append('text')
            .attr('class', 'highlightLabel sizeLabel')
    }

    protected _wrangle(data: BarChartData) {
        return data;
    }

    protected _render(rd: BarChartData): void {
        const op = this.options;

        const xScale = d3.scaleLinear().domain([0, rd.values.length])
            .range([5, op.width - 35]);

        const extent = rd.extent || [0, d3.max(rd.values)]
        const yScale = d3.scaleLinear().domain(extent)
            .nice(10).range([op.height - 35, op.margin_top]);

        const adjustWidth = (bandH: number) => (bandH > 5) ? (bandH - 1) : (0.9 * bandH);
        const width = adjustWidth(xScale(1) - xScale(0))

        const colorValue = (index: number) => rd.colors ? (rd.colors[index % rd.colors.length]) : null;


        const allData = rd.values.map((v, i) => ({ v, c: colorValue(i) }))

        this.layers.main.selectAll('.bar').data(allData)
            .join('rect')
            .attr('class', 'bar')
            .attrs({
                x: (d, i) => xScale(i),
                y: d => yScale(d.v),
                width: d => width,
                height: d => op.height - 35 - yScale(d.v),
            })
            .style('fill', d => d.c)
            .style('opacity', 1)
            .on('mouseenter', (d,i) => {
                const x = xScale(i) + .5 * width;
                const y = yScale(d.v) - 2;

                this.highlightLabel
                    .attr('transform', `translate(${x},${y})`)
                    .style('visibility', null)
                    .text(() => d.v);
            })
            .on('mouseleave', () => this.highlightLabel.style('visibility', 'hidden'))

        this.layers.bg.select('.y-axis').call(d3.axisRight(yScale).tickFormat(op.numberFormat));


    }


}