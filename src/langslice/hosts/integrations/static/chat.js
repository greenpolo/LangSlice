'use strict';
const $=id=>document.getElementById(id);
const conversation=$('conversation'), messages=$('messages');
let sequence=0, active=null, follow=true, imageFollow=true, batches=[], batchIndex=-1;
const tools=new Map(), pendingTools=new Map();
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function inline(s){return esc(s).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>');}
function markdown(text){
  const out=[];let para=[],list=null,code=null;
  function flush(){if(para.length){out.push('<p>'+para.map(inline).join('<br>')+'</p>');para=[];}if(list){out.push('</'+list+'>');list=null;}}
  for(const line of text.split('\n')){
    if(line.startsWith('```')){flush();if(code!==null){out.push('<pre><code>'+esc(code.join('\n'))+'</code></pre>');code=null;}else code=[];continue;}
    if(code!==null){code.push(line);continue;}
    if(!line.trim()){flush();continue;}
    const head=line.match(/^#{1,6}\s+(.+)/);if(head){flush();out.push('<h3>'+inline(head[1])+'</h3>');continue;}
    const item=line.match(/^\s*(?:([-*])|\d+\.)\s+(.+)/);if(item){if(para.length)flush();const tag=item[1]?'ul':'ol';if(list!==tag){if(list)out.push('</'+list+'>');out.push('<'+tag+'>');list=tag;}out.push('<li>'+inline(item[2])+'</li>');continue;}
    if(list)flush();if(line.startsWith('> ')){flush();out.push('<blockquote>'+inline(line.slice(2))+'</blockquote>');}else para.push(line);
  }flush();if(code!==null)out.push('<pre><code>'+esc(code.join('\n'))+'</code></pre>');return out.join('');
}
function labelTarget(ids){return(ids||[]).slice(0,5).map(s=>String(s).replace(/^section_0*(\d+)\.[^.]+$/,'Slice $1')).join(', ')+((ids||[]).length>5?' …':'');}
function title(name){return String(name||'tool');}
function note(text,cls='notice'){const el=document.createElement('div');el.className=cls;el.textContent=text;messages.append(el);return el;}
function toolCard(event){
  const key=event.execution_id||event.id||event.name;
  let card=tools.get(key);if(card)return card;
  const root=document.createElement('details');root.className='tool running';
  root.innerHTML='<summary><span class="tool-copy"><span class="tool-title"></span><span class="tool-target"></span></span><span class="tool-state">[running]</span></summary><div class="tool-detail"></div>';
  root.querySelector('.tool-title').textContent=title(event.name);
  root.querySelector('.tool-target').textContent=labelTarget(event.target_ids);
  const detail=root.querySelector('.tool-detail');
  if(event.args&&Object.keys(event.args).length){const heading=document.createElement('h4');heading.textContent='Tool inputs';const pre=document.createElement('pre');pre.textContent=JSON.stringify(event.args,null,2);detail.append(heading,pre);}
  card={root,detail,key,name:event.name};tools.set(key,card);if(event.id)tools.set(event.id,card);
  const pending=pendingTools.get(event.name)||[];pending.push(card);pendingTools.set(event.name,pending);
  messages.append(root);return card;
}
function resultCard(card,event){
  card.root.classList.remove('running');
  const response=event.response||{};const bad=response.status&& !['ok','success','submitted'].includes(String(response.status).toLowerCase());
  card.root.classList.toggle('failed',Boolean(bad));
  card.root.querySelector('.tool-state').textContent=bad?'['+String(response.status)+']':'[done]';
  let pre=card.detail.querySelector('.result-payload');
  if(!pre){const heading=document.createElement('h4');heading.textContent='Tool result';pre=document.createElement('pre');pre.className='result-payload';card.detail.append(heading,pre);}
  pre.textContent=JSON.stringify(response,null,2);
}
function showImages(){
  const grid=$('image-grid');grid.replaceChildren();const batch=batches[batchIndex];
  $('image-count').textContent=batch?`${batchIndex+1} / ${batches.length}`:'0 / 0';
  $('follow-images').classList.toggle('active',imageFollow);
  if(!batch){$('image-label').textContent='No images received';const empty=document.createElement('div');empty.className='image-empty';empty.textContent='No tool images in this session.';grid.append(empty);return;}
  $('image-label').textContent=batch.label;grid.classList.toggle('single',batch.images.length===1);
  // Four images fill the compact tray. Batches larger than four become pages.
  for(const entry of batch.images){const tile=document.createElement('div');tile.className='image-tile';const img=document.createElement('img');img.src='image/'+entry.id;img.alt=entry.label;const caption=document.createElement('span');caption.textContent=entry.label;tile.append(img,caption);img.onerror=()=>{caption.textContent='Image no longer in live history';};tile.onclick=()=>{$('large-image').src=img.src;$('large-label').textContent=entry.label;$('lightbox').showModal();};grid.append(tile);}
}
function imageEvent(event,card=null){
  if(!event.images?.length)return;
  const pages=[];for(let i=0;i<event.images.length;i+=4){const batch={images:event.images.slice(i,i+4),label:event.name?title(event.name):'Starting images'};batches.push(batch);pages.push(batch);}
  while(batches.length>40)batches.shift();
  if(imageFollow)batchIndex=batches.length-pages.length;else batchIndex=Math.max(0,Math.min(batchIndex,batches.length-1));
  const button=document.createElement('button');button.className='image-event';button.innerHTML='<span></span>';
  button.lastChild.textContent=`[View ${event.images.length} image${event.images.length===1?'':'s'}]`;
  button.onclick=()=>{const index=batches.indexOf(pages[0]);if(index>=0){imageFollow=false;batchIndex=index;showImages();}};
  if(card)card.root.after(button);else messages.append(button);showImages();
}
function render(event){
  $('welcome').hidden=true;
  if(event.kind==='text'||event.kind==='reasoning'){
    if(!active||active.kind!==event.kind){const root=document.createElement('article');root.className='message '+(event.kind==='text'?'assistant':'reasoning');root.innerHTML='<div class="message-label"><span></span></div><div class="message-body"></div>';root.querySelector('.message-label').lastChild.textContent=event.kind==='reasoning'?'Reasoning summary':'Agent';messages.append(root);active={kind:event.kind,text:'',body:root.querySelector('.message-body')};}
    active.text=event.revised?event.text:active.text+(event.text||'');
    if(active.text.length>180000){active.text=active.text.slice(-180000);active.trimmed=true;}
    active.body.innerHTML=(active.trimmed?'<p class="notice">Earlier text was removed from this live message.</p>':'')+markdown(active.text);return;
  }
  if(event.kind!=='status')active=null;
  if(event.kind==='tool_start')toolCard(event);
  else if(event.kind==='tool_end'){const card=toolCard(event);resultCard(card,event);imageEvent(event,card);}
  else if(event.kind==='tool_result'){
    const pending=pendingTools.get(event.name)||[];
    const card=tools.get(event.execution_id||event.id)||pending.find(c=>!c.received);
    if(card){card.received=true;resultCard(card,event);const index=pending.indexOf(card);if(index>=0)pending.splice(index,1);}
    imageEvent(event,card);
  }
  else if(event.kind==='seed')imageEvent(event);
  else if(event.kind==='complete')note(event.submitted?'Registration submitted to ABBA.':'Agent stopped before submission.');
  else if(event.kind==='error')note(event.text,'notice error');
}
conversation.addEventListener('scroll',()=>{follow=conversation.scrollHeight-conversation.scrollTop-conversation.clientHeight<70;$('jump').hidden=follow;});
$('jump').onclick=()=>{follow=true;conversation.scrollTop=conversation.scrollHeight;$('jump').hidden=true;};
$('previous').onclick=()=>{imageFollow=false;batchIndex=Math.max(0,batchIndex-1);showImages();};
$('next').onclick=()=>{imageFollow=false;batchIndex=Math.min(batches.length-1,batchIndex+1);showImages();};
$('follow-images').onclick=()=>{imageFollow=true;batchIndex=batches.length-1;showImages();};
$('close-image').onclick=()=>$('lightbox').close();
async function poll(){
  try{const response=await fetch('events?since='+sequence,{cache:'no-store'});if(!response.ok)throw new Error('unavailable');const state=await response.json();
    if(state.reset){sequence=0;active=null;messages.replaceChildren();tools.clear();pendingTools.clear();batches=[];batchIndex=-1;showImages();return;}
    $('status').textContent=state.status;$('connection').textContent='Connected locally';
    if(!sequence&&state.trimmed)note('Earlier activity was removed from the live history.');
    const stick=follow;for(const event of state.events)render(event);sequence=state.sequence;
    // Bound the visible transcript too; server replay bounds do not constrain a long-open tab.
    while(messages.children.length>600){const old=messages.firstChild;old.remove();for(const [key,card] of tools)if(card.root===old)tools.delete(key);for(const [name,cards] of pendingTools)pendingTools.set(name,cards.filter(c=>c.root!==old));}
    if(stick)conversation.scrollTop=conversation.scrollHeight;
  }catch{ $('connection').textContent='Waiting for connection…'; }
  finally{setTimeout(poll,180);}
}
poll();
