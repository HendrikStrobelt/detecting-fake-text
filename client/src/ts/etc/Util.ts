import * as d3 from "d3";

/**
 * Created by hen on 5/15/17.
 */
let the_unique_id_counter = 0;

export class Util {
    static simpleUId({prefix = ''}): string {
        the_unique_id_counter += 1;

        return prefix + the_unique_id_counter;
    }
}

export type D3Sel = d3.Selection<any, any, any, any>

export function argsort(array, sortFct):number[] {
    return array
        .map((d, i) => [d, i])
        .sort((a,b) => sortFct(a[0], b[0]))
        .map(d => d[1]);
}

export function range(end){
    return [...Array(end).keys()]
}

export function obj_to_arr(obj:object){
    const sortedKeys = Object.keys(obj).sort();
    const res=[];
    sortedKeys.forEach(k => {res.push(k); res.push(obj[k])})
    return res;
}

export function arr_to_obj(arr:any){
    const res={};
    const max_l = Math.floor(arr.length/2);
    for (let i = 0; i<max_l; i++){
        res[arr[2*i]] = arr[2*i+1];
    }
    return res;
}

export function splitString(string, splitters) {
    var list = [string];
    for(var i=0, len=splitters.length; i<len; i++) {
        traverseList(list, splitters[i], 0);
    }
    return flatten(list);
}

export function traverseList(list, splitter, index) {
    if(list[index]) {
        if((list.constructor !== String) && (list[index].constructor === String))
            (list[index] != list[index].split(splitter)) ? list[index] = list[index].split(splitter) : null;
        (list[index].constructor === Array) ? traverseList(list[index], splitter, 0) : null;
        (list.constructor === Array) ? traverseList(list, splitter, index+1) : null;
    }
}

export function flatten(arr) {
    return arr.reduce(function(acc, val) {
        return acc.concat(val.constructor === Array ? flatten(val) : val);
    },[]);
}

export function token_cleanup(token) {

    token = (token.startsWith('Ġ')) ? token.slice(1) : ((token.startsWith('Ċ') || token.startsWith('â')) ? " " : token);
    // token = (token.startsWith('â')) ? '–' : token;
    // token = (token.startsWith('ľ')) ? '“' : token;
    // token = (token.startsWith('Ŀ')) ? '”' : token;
    // token = (token.startsWith('Ļ')) ? "'" : token;

    try {
        token = decodeURIComponent(escape(token));
    } catch{
        console.log(token, '-- token is hard');
    }
    return token;
}