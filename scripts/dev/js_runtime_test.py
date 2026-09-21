"""Dev check: run the dashboard/snapshot page's JavaScript in QuickJS with a stub DOM (there is no browser on the GPU box).

    uv pip install quickjs --target <DIR>
    python scripts/dashboard.py --snapshot out.html
    python scripts/dev/js_runtime_test.py <DIR> out.html

Catches runtime errors (missing variables, bad state handling) that a syntax parse cannot; it does not draw anything.
"""
import sys,re; sys.path.insert(0,sys.argv[1]+"/qk")
import quickjs
h=open(sys.argv[2]).read()
scripts=re.findall(r"<script>(.*?)</script>",h,re.S)
scripts[1]=scripts[1].replace('catch(e){$("status").textContent="dashboard offline"','catch(e){window.LASTERR=String(e&&e.stack||e);$("status").textContent="dashboard offline"')
stub=r'''
var window=this, els={};
function El(id){this.id=id;this.dataset={};this.style={};this.hidden=false;this.value="";this.children=[];this.textContent="";this._h="";this.clientWidth=400;this.clientHeight=190;
 this.getContext=()=>new Proxy({},{get:(t,k)=>k in t?t[k]:()=>{},set:(t,k,v)=>{t[k]=v;return true}});}
Object.defineProperty(El.prototype,"innerHTML",{get(){return this._h},set(v){this._h=v;this.children=v?[1]:[]}});
El.prototype.append=function(){};
var document={getElementById:id=>els[id]||(els[id]=new El(id)),createElement:()=>new El("x"),documentElement:{}};
var getComputedStyle=()=>({getPropertyValue:()=>"#123456"});
var devicePixelRatio=1, addEventListener=()=>{}, setInterval=()=>{}, location={};
function URL(u){this.pathname=String(u).replace("http://x","").split("?")[0]}
'''
ctx=quickjs.Context(); ctx.eval(stub)
ctx.eval("document.getElementById('xaxis').value='time';document.getElementById('smooth').value='0.6';")
for sc in scripts: ctx.eval(sc)
for _ in range(3000):
    if not ctx.execute_pending_job(): break
print("LASTERR:",ctx.eval("window.LASTERR"))
for k in ("status","prog","plannote"):
    print(k,"=>",(ctx.eval(f"(els.{k}||{{}}).textContent") or "")[:200])
print("chart canvases:",ctx.eval("Object.keys(els).filter(k=>/^c\\d$/.test(k)).length"),"| plan rows:",ctx.eval("String((els.plantable||{})._h).split('<tr>').length-1"))
ctx.eval("document.getElementById('xaxis').value='step'; render()"); print("re-render OK")
