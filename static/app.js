const $=id=>document.getElementById(id);
let accessToken=localStorage.getItem('eidomira_access_token')||'';
if(!accessToken)location.replace('/?signin=1');
function apiFetch(url,options={}){const headers=new Headers(options.headers||{});if(accessToken)headers.set('Authorization','Bearer '+accessToken);return fetch(url,{...options,headers})}
function setToken(token){accessToken=token||'';if(token)localStorage.setItem('eidomira_access_token',token);else localStorage.removeItem('eidomira_access_token')}
let authMode='register';
async function loadAccount(){if(!accessToken){location.replace('/?signin=1');return}try{const me=await apiFetch('/api/auth/me').then(r=>{if(!r.ok)throw Error();return r.json()});$('accountBtn').textContent='Account';$('accountBar').hidden=false;$('accountEmail').textContent=me.email;const billing=await apiFetch('/api/billing/account').then(r=>r.json());$('accountPlan').textContent=(billing.subscription?.plan||'NO PLAN').toUpperCase();$('creditBalance').textContent=billing.wallet?.total||0}catch{setToken('');location.replace('/?signin=1')}}
function openAuth(){authMode='register';renderAuth();$('authModal').hidden=false}
function renderAuth(){$('authTitle').textContent=authMode==='register'?'Start your free trial':'Welcome back';$('authCopy').textContent=authMode==='register'?'Verify your email to receive 100 credits for 7 days. No card required.':'Sign in to your Eidomira account.';$('authSubmit').textContent=authMode==='register'?'Create account':'Sign in';$('authSwitch').textContent=authMode==='register'?'Already registered? Sign in':'New to Eidomira? Start free trial';$('authPassword').autocomplete=authMode==='register'?'new-password':'current-password';$('authMessage').textContent=''}
$('accountBtn').onclick=()=>accessToken?$('accountBar').toggleAttribute('hidden'):openAuth();$('authClose').onclick=()=>$('authModal').hidden=true;$('authSwitch').onclick=()=>{authMode=authMode==='register'?'login':'register';renderAuth()};$('logoutBtn').onclick=async()=>{try{await fetch('/api/auth/logout',{method:'POST'})}catch{}setToken('');location.href='/'};
async function startCheckout(product){if(!accessToken)return openAuth();try{const r=await apiFetch('/api/payments/paystack/checkout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product})}),j=await r.json();if(!r.ok)throw Error(j.error||'Checkout unavailable');location.href=j.authorization_url}catch(e){status(e.message,true)}}
$('upgradeBtn').onclick=()=>startCheckout('live-pro-monthly');$('annualBtn').onclick=()=>startCheckout('live-pro-annual');$('paygBtn').onclick=()=>startCheckout($('paygPack').value);document.querySelectorAll('.priceSignup').forEach(b=>b.onclick=openAuth);document.querySelectorAll('.benefitTabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.benefitTabs button').forEach(x=>x.classList.remove('active'));b.classList.add('active')});
$('authForm').onsubmit=async e=>{e.preventDefault();$('authSubmit').disabled=true;try{const r=await fetch('/api/auth/'+authMode,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:$('authEmail').value,password:$('authPassword').value})}),j=await r.json();if(!r.ok)throw Error(j.error||'Authentication failed');if(j.access_token){setToken(j.access_token);$('authModal').hidden=true;await loadAccount()}else $('authMessage').textContent=j.message}catch(err){$('authMessage').textContent=err.message;$('authMessage').style.color='#fda4af'}finally{$('authSubmit').disabled=false}};
async function verifyFromLink(){const token=new URLSearchParams(location.search).get('verify');if(!token)return;history.replaceState({},'',location.pathname);const r=await fetch('/api/auth/verify-email',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})}),j=await r.json();if(r.ok){setToken(j.access_token);status('Email verified. Your 7-day trial is active.');loadAccount()}else{openAuth();$('authMessage').textContent=j.error}}
async function verifyPaymentReturn(){const p=new URLSearchParams(location.search),reference=p.get('reference');if(p.get('payment')!=='return'||!reference)return;history.replaceState({},'',location.pathname);if(!accessToken)return openAuth();status('Confirming Paystack payment…');try{const r=await apiFetch('/api/payments/paystack/verify/'+encodeURIComponent(reference)),j=await r.json();if(!r.ok)throw Error(j.error||'Payment verification failed');if(j.status==='success'){status('Payment confirmed. Live Pro is active.');await loadAccount()}else status('Payment is '+j.status+'. Your plan has not been changed.',true)}catch(e){status(e.message,true)}}
loadAccount();verifyFromLink();verifyPaymentReturn();
let media=null,pc=null,session=null,running=false,statsTimer=null,reconnects=0;
let recorder=null,recordedChunks=[],recordUrl=null,recordedBlob=null,recordStarted=0,recordTimer=null;
let cameraFacing='user',micStream=null,wakeLock=null,deferredInstall=null;
let callRoomInstance=null,callVideoTrack=null,callAudioTrack=null;

async function health(){
  try{const j=await apiFetch('/api/health').then(r=>r.json());
    $('engineName').textContent=j.backend.toUpperCase();
    $('healthText').textContent=j.gpu?((j.provider?j.provider.toUpperCase():'GPU')+((j.accelerated===false)?' · CPU':' neural engine ready')):'Diagnostic transport mode';
    $('healthDot').style.background=(j.gpu&&j.accelerated!==false)?'#6ee7a5':(j.gpu?'#52d3ff':'#f59e0b');
  }catch{$('healthText').textContent='Engine unavailable'}
}
health();

$('source').onchange=e=>{const f=e.target.files[0];if(f){$('sourcePreview').src=URL.createObjectURL(f);$('uploadCard').classList.add('hasImage')}};
$('enrollBtn').onclick=async()=>{
  const f=$('source').files[0];
  if(!f)return status('Choose a reference portrait.',true);
  if(!$('consent').checked)return status('Confirm self-only consent first.',true);
  const d=new FormData();d.append('image',f);d.append('consent','true');d.append('quality',document.querySelector('input[name="quality"]:checked').value);
  status('Analyzing identity…');$('enrollBtn').disabled=true;
  try{const r=await apiFetch('/api/sessions',{method:'POST',body:d}),j=await r.json();
    if(!r.ok)throw Error(j.error||'Enrollment failed');
    session=j.session_id;status('Identity enrolled. Start your camera.');
    $('p1').classList.add('active');$('p2').classList.add('active');
    if(media)$('goBtn').disabled=false;
  }catch(e){status(e.message,true)}finally{$('enrollBtn').disabled=false}
};
function status(t,error=false){$('status').textContent=t;$('status').style.color=error?'#fda4af':'#86efac'}

$('cameraBtn').onclick=async()=>{
  if(media){stopCamera();return}
  try{
    media=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1920,max:1920},height:{ideal:1080,max:1080},frameRate:{ideal:30,max:30},facingMode:{ideal:cameraFacing}},audio:false});
    $('video').srcObject=media;await $('video').play();
    $('cameraEmpty').style.display='none';$('cameraBadge').textContent='LIVE';$('cameraBadge').className='badge live';
    $('cameraBtn').textContent='Stop camera';$('flipBtn').disabled=false;$('p2').classList.add('active');$('goBtn').disabled=!session;
  }catch(e){status('Camera blocked: '+e.message,true)}
};
function stopCamera(){stopTransform();media?.getTracks().forEach(t=>t.stop());media=null;$('video').srcObject=null;$('cameraEmpty').style.display='grid';$('cameraBadge').textContent='OFFLINE';$('cameraBadge').className='badge off';$('cameraBtn').innerHTML='Start camera <b>→</b>';$('goBtn').disabled=true;$('flipBtn').disabled=true}
$('flipBtn').onclick=async()=>{const wasRunning=running;if(wasRunning)stopTransform();media?.getTracks().forEach(t=>t.stop());media=null;cameraFacing=cameraFacing==='user'?'environment':'user';await $('cameraBtn').click();status(cameraFacing==='user'?'Front camera selected.':'Rear camera selected.');};

$('goBtn').onclick=()=>running?stopTransform():startTransform();
async function waitForIce(connection){
  if(connection.iceGatheringState==='complete')return;
  await new Promise(resolve=>{const check=()=>{if(connection.iceGatheringState==='complete'){connection.removeEventListener('icegatheringstatechange',check);resolve()}};connection.addEventListener('icegatheringstatechange',check);setTimeout(resolve,4000)});
}
async function startTransform(){
  if(!session||!media)return;
  $('goBtn').disabled=true;status('Creating secure WebRTC session…');
  try{
    const configResponse=await apiFetch('/api/webrtc/'+session+'/configuration');
    const rtcConfig=await configResponse.json();
    if(!configResponse.ok)throw Error(rtcConfig.error||'Could not obtain ICE configuration');
    pc=new RTCPeerConnection(rtcConfig);
    media.getVideoTracks().forEach(track=>{const transceiver=pc.addTransceiver(track,{direction:'sendrecv'});try{const codecs=RTCRtpSender.getCapabilities('video').codecs;const rank=c=>c.mimeType.toLowerCase()==='video/h264'?0:c.mimeType.toLowerCase()==='video/vp8'?1:2;transceiver.setCodecPreferences([...codecs].sort((a,b)=>rank(a)-rank(b)))}catch{}});
    pc.ontrack=e=>{const stream=e.streams[0]||new MediaStream([e.track]);$('output').srcObject=stream;$('output').style.display='block';$('outputEmpty').style.display='none';$('recordBtn').disabled=false;$('cleanBtn').disabled=false;$('output').play().catch(()=>{})};
    pc.ondatachannel=e=>{if(e.channel.label==='eidomira-telemetry')e.channel.onmessage=handleTelemetry};
    pc.onconnectionstatechange=()=>{
      const state=pc?.connectionState;
      if(state==='connected'){status('Live neural channel connected.');reconnects=0}
      if(['failed','disconnected'].includes(state)&&running)recoverConnection();
    };
    const offer=await pc.createOffer();await pc.setLocalDescription(offer);await waitForIce(pc);
    const response=await apiFetch('/api/webrtc/'+session+'/offer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
    const answer=await response.json();if(!response.ok)throw Error(answer.error||'WebRTC negotiation failed');
    await pc.setRemoteDescription(answer);
    try{wakeLock=await navigator.wakeLock?.request('screen')}catch{}
    running=true;$('goBtn').disabled=false;$('goBtn').innerHTML='<span>■</span> Stop transformation';
    $('outputBadge').textContent='CONNECTING';$('outputBadge').className='badge live';$('p3').classList.add('active');
    statsTimer=setInterval(updateWebRTCStats,1000);
  }catch(e){stopTransform();status('WebRTC error: '+e.message,true);$('goBtn').disabled=false}
}
function handleTelemetry(event){
  const j=JSON.parse(event.data);
  if(j.type==='liveness'){$('livenessPrompt').style.display=j.complete?'none':'block';$('livenessText').textContent=j.instruction;$('livenessBar').style.width=Math.round(j.progress*100)+'%';$('verified').textContent=j.complete?'MATCHING':'LIVE CHECK';if(j.expired)status(j.instruction,true)}
  if(j.type==='verification'){$('livenessPrompt').style.display='none';$('verified').textContent=j.verified?'VERIFIED':'FAILED';if(!j.verified)status('Live face does not match the enrolled identity.',true)}
  if(j.type==='metrics'){$('latency').textContent=j.inference_ms+' ms';$('frameCost').textContent=(j.frame_ms!=null?j.frame_ms+' ms':'— ms');$('fps').textContent=j.fps+' FPS';$('dropped').textContent=j.dropped;$('inferenceSize').textContent=j.inference_width+' px';$('outputBadge').textContent=j.face_found?'LIVE':'NO FACE'}
  if(j.type==='error')status(j.message,true);
}
async function updateWebRTCStats(){
  if(!pc)return;let inbound=0;
  (await pc.getStats()).forEach(r=>{if(r.type==='inbound-rtp'&&r.kind==='video')inbound=r.framesPerSecond||inbound});
  if(inbound)$('fps').textContent=Math.round(inbound)+' FPS';
}
async function recoverConnection(){
  if(reconnects++>=2){status('Connection lost. Press Start transformation to retry.',true);stopTransform();return}
  status('Reconnecting live channel…');stopTransform(false);setTimeout(startTransform,700*reconnects);
}
function stopTransform(reset=true){
  if(recorder&&recorder.state==='recording')stopRecording();
  running=false;clearInterval(statsTimer);statsTimer=null;
  if(pc){pc.onconnectionstatechange=null;pc.getSenders().forEach(s=>{if(s.track)s.replaceTrack(null).catch(()=>{})});pc.close();pc=null}
  wakeLock?.release().catch(()=>{});wakeLock=null;
  $('output').srcObject=null;$('output').style.display='none';$('outputEmpty').style.display='grid';$('livenessPrompt').style.display='none';
  $('goBtn').innerHTML='<span>✦</span> Start transformation';$('outputBadge').textContent='WAITING';$('outputBadge').className='badge';$('recordBtn').disabled=true;$('cleanBtn').disabled=true;exitCleanMode();
  if(reset)reconnects=0;
}

function preferredMime(){
  const choices=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];
  return choices.find(t=>MediaRecorder.isTypeSupported(t))||'';
}
$('recordBtn').onclick=()=>recorder&&recorder.state==='recording'?stopRecording():startRecording();
async function startRecording(){
  const outputStream=$('output').srcObject;if(!outputStream)return status('Start transformation before recording.',true);
  recordedChunks=[];recordedBlob=null;if(recordUrl){URL.revokeObjectURL(recordUrl);recordUrl=null}$('recordDownload').hidden=true;$('shareBtn').hidden=true;
  try{const tracks=[...outputStream.getVideoTracks()];if($('recordMic').checked){micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});tracks.push(...micStream.getAudioTracks())}const stream=new MediaStream(tracks);const mime=preferredMime();const options={videoBitsPerSecond:6_000_000,audioBitsPerSecond:128_000};if(mime)options.mimeType=mime;recorder=new MediaRecorder(stream,options);
    recorder.ondataavailable=e=>{if(e.data.size)recordedChunks.push(e.data)};
    recorder.onstop=finishRecording;recorder.start(1000);recordStarted=Date.now();
    $('recordBtn').textContent='■ Stop 00:00';$('recordBtn').classList.add('recording');
    recordTimer=setInterval(()=>{const s=Math.floor((Date.now()-recordStarted)/1000);$('recordBtn').textContent=`■ Stop ${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`},1000);
    status('Recording locally in your browser.');
  }catch(e){status('Recording unavailable: '+e.message,true)}
}
function stopRecording(){if(recorder&&recorder.state==='recording')recorder.stop();micStream?.getTracks().forEach(t=>t.stop());micStream=null;clearInterval(recordTimer);recordTimer=null;$('recordBtn').textContent='● Record';$('recordBtn').classList.remove('recording')}
function finishRecording(){
  if(!recordedChunks.length)return;
  recordedBlob=new Blob(recordedChunks,{type:recorder.mimeType||'video/webm'});recordUrl=URL.createObjectURL(recordedBlob);
  $('recordDownload').href=recordUrl;$('recordDownload').download=`eidomira-${new Date().toISOString().replace(/[:.]/g,'-')}.webm`;$('recordDownload').hidden=false;$('recordDownload').textContent=`Download recorded clip · ${(recordedBlob.size/1048576).toFixed(1)} MB`;$('shareBtn').hidden=!navigator.share;status('Recording ready. It was not uploaded to the server.');
}
$('shareBtn').onclick=async()=>{if(!recordedBlob)return;const file=new File([recordedBlob],'eidomira-live.webm',{type:recordedBlob.type});try{if(navigator.canShare?.({files:[file]}))await navigator.share({title:'Eidomira Live',text:'Synthetic video created with Eidomira.',files:[file]});else await navigator.share({title:'Eidomira Live',url:recordUrl})}catch(e){if(e.name!=='AbortError')status('Sharing failed: '+e.message,true)}};
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();deferredInstall=e;$('installBtn').hidden=false});
$('installBtn').onclick=async()=>{if(!deferredInstall)return;deferredInstall.prompt();await deferredInstall.userChoice;deferredInstall=null;$('installBtn').hidden=true};
if('serviceWorker' in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('/static/sw.js').catch(()=>{}));
$('cleanBtn').onclick=()=>document.body.classList.contains('cleanMode')?exitCleanMode():enterCleanMode();
function enterCleanMode(){if(!$('output').srcObject)return;document.body.classList.add('cleanMode');document.documentElement.requestFullscreen?.().catch(()=>{});status('Clean output enabled. Press Escape to exit.')}
function exitCleanMode(){document.body.classList.remove('cleanMode');if(document.fullscreenElement)document.exitFullscreen?.().catch(()=>{})}
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement?.tagName))return;if(e.key==='Escape')exitCleanMode();if(e.key.toLowerCase()==='r'&&!$('recordBtn').disabled)$('recordBtn').click();if(e.key.toLowerCase()==='o'&&!$('cleanBtn').disabled)$('cleanBtn').click()});
$('resetBtn').onclick=async()=>{stopTransform();if(session)await apiFetch('/api/sessions/'+session,{method:'DELETE'});session=null;$('source').value='';$('uploadCard').classList.remove('hasImage');$('goBtn').disabled=true;$('verified').textContent='PENDING';status('Session cleared. Choose another portrait.')};
async function requestCallToken(endpoint,payload){const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),j=await r.json();if(!r.ok)throw Error(j.error||'Call request failed');return j}
$('createCall').onclick=async()=>{try{const name=$('callName').value.trim();if(!name)throw Error('Enter a display name');const auth=await requestCallToken('/api/calls/rooms',{display_name:name});$('callRoom').value=auth.room;await connectCall(auth)}catch(e){$('callStatus').textContent=e.message}};
$('joinCall').onclick=async()=>{try{const name=$('callName').value.trim(),room=$('callRoom').value.trim();if(!name||!room)throw Error('Enter your name and room code');const auth=await requestCallToken('/api/calls/join',{display_name:name,room});await connectCall(auth)}catch(e){$('callStatus').textContent=e.message}};
async function connectCall(auth){
  if(!window.LivekitClient)throw Error('LiveKit client could not load');
  const processed=$('output').srcObject?.getVideoTracks()[0];if(!processed)throw Error('Start transformation before joining a call');
  if(callRoomInstance)await leaveCall();
  const {Room,RoomEvent,LocalVideoTrack,createLocalAudioTrack}=LivekitClient;callRoomInstance=new Room({adaptiveStream:true,dynacast:true});
  callRoomInstance.on(RoomEvent.TrackSubscribed,(track)=>{const el=track.attach();el.autoplay=true;$('remoteGrid').appendChild(el)});
  callRoomInstance.on(RoomEvent.TrackUnsubscribed,track=>track.detach().forEach(el=>el.remove()));
  callRoomInstance.on(RoomEvent.Disconnected,()=>{$('callStatus').textContent='Call disconnected';resetCallUI()});
  await callRoomInstance.connect(auth.url,auth.token);callVideoTrack=new LocalVideoTrack(processed);await callRoomInstance.localParticipant.publishTrack(callVideoTrack,{name:'eidomira-output'});
  try{callAudioTrack=await createLocalAudioTrack({echoCancellation:true,noiseSuppression:true});await callRoomInstance.localParticipant.publishTrack(callAudioTrack)}catch{}
  $('callStatus').textContent=`Connected · ${auth.room}`;$('leaveCall').hidden=false;$('createCall').hidden=true;$('joinCall').hidden=true;
}
$('leaveCall').onclick=leaveCall;
async function leaveCall(){callAudioTrack?.stop();callAudioTrack=callVideoTrack=null;if(callRoomInstance){await callRoomInstance.disconnect();callRoomInstance=null}$('remoteGrid').replaceChildren();$('callStatus').textContent='Not connected';resetCallUI()}
function resetCallUI(){$('leaveCall').hidden=true;$('createCall').hidden=false;$('joinCall').hidden=false}
window.addEventListener('beforeunload',()=>{if(recorder?.state==='recording')recorder.stop();if(recordUrl)URL.revokeObjectURL(recordUrl);micStream?.getTracks().forEach(t=>t.stop());media?.getTracks().forEach(t=>t.stop());wakeLock?.release();callRoomInstance?.disconnect();pc?.close()});
