import * as d3 from 'd3';
import "d3-selection-multi";

import '../css/start.scss'
import "!file-loader?name=index.html!../index.html";
import {SimpleEventHandler} from "./etc/SimpleEventHandler";
import {AnalyzeResponse, GLTR_API} from "./api/GLTR_API";
import {GLTR_HoverEvent, GLTR_Mode, GLTR_Text_Box} from "./vis/GLTR_Text_Box";
import {ToolTip} from "./vis/ToolTip";
import URLHandler from "./etc/URLHandler";
import {Histogram} from './vis/Histogram';
import {BarChart, BarChartData} from './vis/BarChart';


const current = {
    sidebar: {
        width: 400,
        visible: false
    },
    demo: true,
    entropyThreshold: 10,
    project_name: 'gpt-2-small'
};


const mapIDtoEnum = {
    mode_topk: GLTR_Mode.topk,
    mode_diff_p: GLTR_Mode.diff_p,
    mode_frac_p: GLTR_Mode.fract_p

}


window.onload = () => {
    const eventHandler = new SimpleEventHandler(<Element>d3.select('body').node());
    if (URLHandler.parameters['nodemo']){
        current.demo = false;


    }


    const side_bar = d3.select(".side_bar");
    side_bar.style('width', `${current.sidebar.width}px`);

    const api_prefix = URLHandler.parameters['api'] || '';
    const api = new GLTR_API(api_prefix);

    const toolTip = new ToolTip(d3.select('#major_tooltip'), eventHandler);

    const submitBtn = d3.select('#submit_text_btn');
    const textField = d3.select('#test_text');
    const all_mode_btn = d3.selectAll('.btn_mode');

    const stats_top_k = new BarChart(d3.select('#stats_top_k'), eventHandler);
    const stats_frac = new Histogram(d3.select('#stats_frac'), eventHandler);
    const stats_entropy = new Histogram(d3.select('#stats_entropy'), eventHandler);

    const currentColorThresholds = () => {

        return [
            +d3.select('#color1').property('value'),
            +d3.select('#color2').property('value'),
            +d3.select('#color3').property('value'),
        ]
    }


    const currentMode = () => {
        const id = all_mode_btn
            .filter(function () {
                return d3.select(this).classed('selected')
            })
            .property('id');
        return mapIDtoEnum[id];
    }

    const lmf = new GLTR_Text_Box(d3.select("#results"), eventHandler, {color_thresholds: currentColorThresholds()});

    // *****************************
    // *****  demo stuff *****
    // *****************************

    const startSystem = () => {
        d3.select('#model_name').text(current.project_name);
        d3.select('#loader').style('opacity', 0);
        d3.selectAll('.main_frame').style('opacity', 1);
    }

    if (current.demo) {

        // d3.json('demo/examples.json').then(
        const all_demos: { file: string, description: string, api: AnalyzeResponse }[] = require('../demo/'+current.project_name+'_examples.json')


        const load_demo = d => {
            updateFromRequest(d.api);
            textField.property('value', d.api.request.text);
        }

        d3.select('.demos').selectAll('.demoBtn').data(all_demos)
            .join('div')
            .attr('class', 'demoBtn')
            .html(d => d.description)
            .on('click', d => {
                submitBtn.classed('inactive', true);
                if (!d.api) {
                    d3.selectAll(".loadersmall").style('display', null);
                    d3.json('demo/' + d.file).then(
                        api_r => {
                            d.api = <AnalyzeResponse>api_r;

                            load_demo(d);
                        }
                    );
                } else {
                    load_demo(d);
                }
            })

        startSystem();
    } else {

        api.all_projects().then(projects => {
            current.project_name = Object.keys(projects)[0];
            d3.selectAll('.demo').remove();
            startSystem();
        })

    }


    // *****************************
    // *****  Update Vis *****
    // *****************************


    const updateTopKstat = () => {
        const u = <BarChartData>lmf.colorStats;

        stats_top_k.update(u);
    }

    const updateEntropyStat = (data: AnalyzeResponse) => {
        const entropies = data.result.pred_topk.map(topK => {
            const allV = topK.slice(0, current.entropyThreshold).map(k => k[1]);

            const sum = allV.reduce((sum, actual) => sum + actual)
            const entropy = -1. * allV
                .map(v => v / sum)
                .map(x => x == 0 ? 0 : x * Math.log(x))
                .reduce((s, a) => s + a, 0);

            return entropy;
        })

        stats_entropy.update({
            data: entropies,
            no_bins: 8
        })


    }

    const updateFromRequest = (data: AnalyzeResponse) => {
        console.log(data, "--- data");

        d3.select('#all_result').style('opacity', 1).style('display', null);
        d3.selectAll(".loadersmall").style('display', 'none');


        lmf.update(data.result);


        updateTopKstat();

        updateEntropyStat(data);
        // stats_top_k.update({
        //     color: "#70b0ff",
        //     detail: data.result.real_topk.map(d => d[0]),//.filter(d => d < 11),
        //     label: "top k labels",
        //     noBins: 10
        // })

        const fracs = data.result.real_topk.map((d, i) => d[1] / (data.result.pred_topk[i][0][1]))

        stats_frac.update({
            data: fracs,
            label: "frac",
            no_bins: 10,
            extent: [0, 1]
        })

        submitBtn.classed('inactive', false);

    }

    submitBtn.on('click', () => {
        const t = textField.property('value');
        d3.selectAll(".loadersmall").style('display', null);
        submitBtn.classed('inactive', true);
        api.analyze(current.project_name, t).then(updateFromRequest)

    });

    // *****************************
    // *****  mode change  *****
    // *****************************


    all_mode_btn
        .on('click', function () {
            const me = this;
            all_mode_btn.classed('selected', function () {
                return this === me
            });
            lmf.updateOptions({gltrMode: currentMode()}, true);
        });


    d3.selectAll('.colorThreshold').on('input', () => {
        lmf.updateThresholdValues(currentColorThresholds());
        updateTopKstat()
    })


    eventHandler.bind(GLTR_Text_Box.events.tokenHovered, (ev: GLTR_HoverEvent) => {
        if (ev.hovered) {
            toolTip.updateData(ev.d);
        } else {
            toolTip.visility = false;
        }
    })


    d3.select('body').on('touchstart', () => {
        toolTip.visility = false;
    })


    const mainWindow = {
        width: () => window.innerWidth - (current.sidebar.visible ? current.sidebar.width : 0),
        height: () => window.innerHeight - 195
    };


    function setup_ui() {


        d3.select('#sidebar_btn')
            .on('click', function () {
                const sb = current.sidebar;

                sb.visible = !sb.visible;
                d3.select(this)
                    .classed('on', sb.visible);
                side_bar.classed('hidden', !sb.visible);
                side_bar.style('right',
                    sb.visible ? null : `-${current.sidebar.width}px`);

                re_layout();
            });


        window.onresize = () => {
            const w = window.innerWidth;
            const h = window.innerHeight;
            // console.log(w, h, "--- w,h");

            re_layout(w, h);


        };

        function re_layout(w = window.innerWidth, h = window.innerHeight) {
            d3.selectAll('.sidenav')
                .style('height', (h - 53) + 'px');

            const sb = current.sidebar;
            const mainWidth = w - (sb.visible ? sb.width : 0);
            d3.selectAll('.main_frame')
                .style('height', (h - 53) + 'px')
                .style('width', mainWidth + 'px')

            // eventHandler.trigger(GlobalEvents.window_resize, {w, h})

            // eventHandler.trigger(GlobalEvents.main_resize, {
            //     w: (w - global.sidebar()),
            //     h: (h - 45)
            // })

        }

        re_layout(window.innerWidth, window.innerHeight);

    }

    setup_ui();
};




