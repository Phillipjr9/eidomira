const $=id=>document.getElementById(id);
let accessToken=localStorage.getItem('eidomira_access_token')||'';
/* No redirect on this line. An empty token store used to send the browser to the sign-in
 * card right here, before anything could put the session back — so a lost session looked
 * like a logout even though the next thing the page does is try to restore it. loadAccount()
 * is the single place that decides, and it decides after that attempt. */
/* The one place a visit ends, and the one place that says why.
 *
 * Three different things used to arrive here — a token the server refused, a request the
 * server refused for a reason that had nothing to do with the session, and a page that never
 * had a token at all — and from the outside all three looked identical. They are not the same
 * thing, and the difference is this whole bug: a refusal aimed at one request must never sign
 * anybody out. The reason is said out loud, because the console is where this gets diagnosed
 * and because the sign-in card is otherwise indistinguishable from a broken login. */
let sessionEnded=false;
function endSession(reason,ended=true){
  if(sessionEnded)return;
  sessionEnded=true;
  console.warn('[eidomira] ending this session: '+reason);
  setToken('');
  location.replace('/?signin=1'+(ended?'&ended=1':''));
}

/* One place where a request is allowed to disagree about the session.
 *
 * A 401 on a call that carried a token used to mean "that token is finished": dropped, and the
 * visitor sent to the sign-in card. That reading is wrong for every endpoint except the one
 * whose job is to identify the token. `/api/billing/account` answers 401 to an anonymous
 * caller, which is a correct refusal — and the page took it for a dead session and signed the
 * visitor out while `/api/auth/me` was happily answering 200 with their account. So a 401 now
 * buys a restore attempt and a retry first, and whichever way that goes, only the session
 * endpoint may end a session. */
async function apiFetch(url,options={}){
  const headers=new Headers(options.headers||{});
  sendToken(headers);
  const response=await fetch(url,{...options,headers});
  // `reauth:false` marks the calls where a 401 is not about the session: a wrong current
  // password is refused with 403 for exactly this reason, and anything else that can answer
  // 401 while the session is perfectly alive says so by passing this.
  const rejected=response.status===401&&accessToken&&options.reauth!==false;
  if(!rejected)return response;

  console.warn('[eidomira] '+url+' answered 401 while a token was sent');
  // Where the demo is on this is almost always the preview's own doing — a rebuilt database or
  // a changed hostname — so put the session back and repeat the request once. The token is
  // deliberately not cleared first: it may be perfectly good, and throwing it away here is
  // what left the session check below with nothing to check.
  if(await restoreDemoSession()){
    const retried=new Headers(options.headers||{});
    sendToken(retried);
    const retry=await fetch(url,{...options,headers:retried});
    if(retry.status!==401)return retry;
    console.warn('[eidomira] '+url+' still answered 401 with a brand-new session');
  }

  // Before signing anybody out, ask the one endpoint whose whole job is to say who this token
  // belongs to. A refusal from anywhere else means "you may not do this", which is not the
  // same as "you are not signed in".
  const user=await sessionUser();
  if(user===undefined){
    // The server could not be asked, so nobody knows. Unknown is never a reason to sign out.
    status('The server did not answer just now. Your session is untouched — try again in a moment.',true);
    return response;
  }
  if(user){
    me=user;
    status('That part of the page was refused, but your session is fine. Reload if it repeats.',true);
    console.warn('[eidomira] keeping the session for '+user.email+' despite the 401 from '+url);
    return response;
  }
  endSession('the server refused this token on '+url);
  return response;
}

/* Ask the server who this token belongs to — the only authority on whether this browser is
 * signed in. Deliberately not routed through apiFetch: this is the call that apiFetch asks
 * when it needs to know, and a circular answer would be worth nothing.
 *
 * It answers three different things and the difference matters:
 *   a record  — this is a session, and this is whose
 *   null      — there is no session: the token was refused, or there was no token to send
 *   undefined — the server could not be asked, which says nothing about the session
 *
 * A 200 is not by itself a session. `/api/auth/me` answers 200 to an anonymous caller with a
 * guest record whose address is local@eidomira.invalid, so a caller that only checked
 * `response.ok` would take that guest for a signed-in user — and the page could then sit there
 * looking signed in as somebody who does not exist. */
async function sessionUser(){
  let response;
  try{
    const headers=new Headers();
    sendToken(headers);
    response=await fetch('/api/auth/me',{headers});
  }catch(error){
    console.warn('[eidomira] could not reach the server to check the session:',error.message);
    return undefined;
  }
  if(response.status===401)return null;          // the token itself was refused
  if(!response.ok){
    console.warn('[eidomira] /api/auth/me answered '+response.status);
    return undefined;
  }
  const user=await response.json();
  if(!user||!user.id||user.role==='guest'||user.id==='local-guest'){
    // Worth saying out loud, because there are two ways to get here and they need different
    // fixes: nothing was ever signed in, or this page is holding a token that did not reach
    // the server. The second one has been seen, and it is invisible from the client.
    console.warn('[eidomira] the server answered 200 as a guest — no credential reached it'+
      (accessToken?' (this page is holding one)':''));
    return null;
  }
  return user;
}

/* Type `await eidomiraSession()` in this page's console to see what the page believes about the
 * session and why. It exists because every one of these failures is a decision this file makes,
 * and that decision used to be invisible from outside. It prints no credential: present or not,
 * when it expires, and what the server says about it. */
window.eidomiraSession=async()=>{
  const user=await sessionUser();
  const expiry=tokenExpiry();
  const report={
    token:accessToken?'present':'none',
    expires:expiry?expiry.toLocaleString():'unknown',
    server:user===undefined?'did not answer':user===null?'no session (guest, or the token was refused)':user.email,
    role:user?user.role:null,
    role_remembered_for_the_demo:(()=>{try{return localStorage.getItem(DEMO_ROLE_KEY)}catch{return"unreadable"}})(),
    restore_already_attempted:restoreAttempted,
    signed_out:sessionEnded,
    storage:Object.keys(localStorage),
  };
  console.log('[eidomira] session',report);
  return report;
};
/* How this page presents the token, and why it is said twice.
 *
 * `Authorization` is the right header and stays the primary one. It also turned out not to
 * survive the hosted preview: every request whose credential travelled in a body arrived
 * intact, and every request carrying a token in `Authorization` reached the server with no
 * credential at all — a guest answer to a browser that was holding one, and a fresh token did
 * not change it. The cookie cannot cover for it either, being SameSite=Lax and therefore
 * unsent from inside a cross-site iframe. So the same token also goes in a plain header that
 * nothing has a reason to consume, and the server takes whichever arrives. Two headers, one
 * credential, no fallback logic: whichever one the deployment preserves is enough.
 */
function sendToken(headers){
  if(!accessToken)return headers;
  headers.set('Authorization','Bearer '+accessToken);
  headers.set('X-Eidomira-Token',accessToken);
  return headers;
}
function setToken(token){accessToken=token||'';if(token)localStorage.setItem('eidomira_access_token',token);else localStorage.removeItem('eidomira_access_token')}
function creditCount(n){return (n||0).toLocaleString('en-NG')}
function walletDate(seconds){return new Date(seconds*1000).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'})}
function note(node,message,bad=false){node.textContent=message||'';node.style.color=bad?'#fda4af':''}

/* The account this session belongs to, kept from /api/auth/me so the billing view can tell
 * whether this browser is verified without fetching it twice. */
let me=null;

/* ------------------------------------------------------------------- sections
 * Three views, one visible at a time, addressed by the URL hash so a link is shareable and
 * the back button works. This is the only navigation: the credit chip and the account chip
 * in the bar point at these views rather than repeating the numbers inside them, so a
 * balance or a plan is rendered in exactly one place.
 */
const VIEWS=['studio','billing','settings'];
const viewId=name=>'view'+name[0].toUpperCase()+name.slice(1);

function showView(name){
  const wanted=VIEWS.includes(name)?name:'studio';
  VIEWS.forEach(v=>{$(viewId(v)).hidden=v!==wanted});
  document.querySelectorAll('.tab[data-view]').forEach(tab=>{
    const on=tab.dataset.view===wanted;
    tab.classList.toggle('active',on);
    if(on)tab.setAttribute('aria-current','page');else tab.removeAttribute('aria-current');
  });
}
addEventListener('hashchange',()=>showView((location.hash||'#studio').slice(1)));

/* ----------------------------------------------------------------- preferences
 * Stored in this browser and applied to the next camera start or enrolment. Deliberately
 * not account settings: they describe this device, and syncing them would mean storing
 * device names on the server for no benefit.
 */
const PREFS_KEY='eidomira_studio_prefs';
const DEFAULT_PREFS={quality:'balanced',cameraId:'',micId:'',reduceMotion:false};
function readPrefs(){
  try{return {...DEFAULT_PREFS,...JSON.parse(localStorage.getItem(PREFS_KEY)||'{}')}}
  catch{return {...DEFAULT_PREFS}}
}
function writePrefs(patch){
  const next={...readPrefs(),...patch};
  try{localStorage.setItem(PREFS_KEY,JSON.stringify(next))}catch{}
  applyPrefs(next);
  return next;
}
function applyPrefs(prefs=readPrefs()){
  document.querySelectorAll('input[name=quality]').forEach(radio=>{radio.checked=radio.value===prefs.quality});
  const label=prefs.quality[0].toUpperCase()+prefs.quality.slice(1);
  $('qualityNow').textContent=label;
  $('motionToggle').checked=Boolean(prefs.reduceMotion);
  document.body.classList.toggle('reduceMotion',Boolean(prefs.reduceMotion));
  if($('cameraDevice'))$('cameraDevice').value=prefs.cameraId||'';
  if($('micDevice'))$('micDevice').value=prefs.micId||'';
}

function fillDeviceSelect(select,label,devices){
  const chosen=select.value;
  select.replaceChildren();
  const fallback=document.createElement('option');
  fallback.value='';fallback.textContent=label;
  select.append(fallback);
  devices.forEach(device=>{
    const option=document.createElement('option');
    option.value=device.deviceId;
    option.textContent=device.label||('Device ending '+(device.deviceId||'').slice(-4));
    select.append(option);
  });
  if([...select.options].some(option=>option.value===chosen))select.value=chosen;
}

async function listDevices(){
  const camera=$('cameraDevice'),mic=$('micDevice'),noteNode=$('deviceNote');
  if(!navigator.mediaDevices?.enumerateDevices){noteNode.textContent='This browser cannot list devices, so the defaults are used.';return}
  try{
    const devices=await navigator.mediaDevices.enumerateDevices();
    fillDeviceSelect(camera,'Default camera',devices.filter(d=>d.kind==='videoinput'));
    fillDeviceSelect(mic,'Default microphone',devices.filter(d=>d.kind==='audioinput'));
    noteNode.textContent=devices.some(d=>d.label)
      ?'Choosing a device applies the next time the camera starts.'
      :'Device names appear once you have allowed camera access at least once in this browser.';
  }catch{
    noteNode.textContent='Devices could not be listed in this browser.';
  }
}

/* ----------------------------------------------------------------------- auth */
let authMode='register';
function renderAuth(){$('authTitle').textContent=authMode==='register'?'Start your free trial':'Welcome back';$('authCopy').textContent=authMode==='register'?'Verify your email to receive 100 credits for 7 days. No card required.':'Sign in to your Eidomira account.';$('authSubmit').textContent=authMode==='register'?'Create account':'Sign in';$('authSwitch').textContent=authMode==='register'?'Already registered? Sign in':'New to Eidomira? Start free trial';$('authPassword').autocomplete=authMode==='register'?'new-password':'current-password';$('authMessage').textContent=''}
function openAuth(mode){authMode=mode==='login'?'login':'register';renderAuth();$('authModal').hidden=false}
$('authClose').onclick=()=>$('authModal').hidden=true;
$('authSwitch').onclick=()=>{authMode=authMode==='register'?'login':'register';renderAuth()};
$('authForm').onsubmit=async e=>{e.preventDefault();$('authSubmit').disabled=true;try{const r=await fetch('/api/auth/'+authMode,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:$('authEmail').value,password:$('authPassword').value})}),j=await r.json();if(!r.ok)throw Error(j.error||'Authentication failed');if(j.access_token){setToken(j.access_token);$('authModal').hidden=true;await loadAccount()}else $('authMessage').textContent=j.message}catch(err){$('authMessage').textContent=err.message;$('authMessage').style.color='#fda4af'}finally{$('authSubmit').disabled=false}};

/* One-click demo sign-in.
 *
 * The password form is this page's real path; the demo row appears only when the server
 * says it is switched on, and it carries no credentials at all — pressing it asks the
 * server for a session, so there is nothing on this page to read, copy or leak.
 */
/* Which demo account this browser signed in as, so a session that is lost can be restored to
 * the same one rather than to whichever button was pressed first. */
const DEMO_ROLE_KEY='eidomira_demo_role';
let demoMethods=null;
let restoreAttempted=false;

async function loadAuthMethods(){
  try{
    const methods=await fetch('/api/auth/methods').then(r=>r.json());
    demoMethods=methods;
    if(!methods.demo_login)return;
    const accounts=methods.demo_accounts||[];
    $('demoRow').querySelectorAll('.demoButton').forEach(button=>{
      const account=accounts.find(a=>a.role===button.dataset.role);
      if(account)button.querySelector('[data-email]').textContent=account.email;
      button.onclick=()=>demoSignIn(button);
    });
    if(methods.demo_password){
      const hint=$('demoHint'),shown=document.createElement('code');
      shown.textContent=methods.demo_password;
      $('demoRow').querySelector('[data-demo-label]').textContent='Demo access · one click, or type the password below';
      hint.replaceChildren('Both accounts accept the same password — ',shown);
      hint.hidden=false;
    }
    $('demoRow').hidden=false;
  }catch{/* a demo is a convenience; the form is the product */}
}

/* Sign in as a demo account — through the password, on the ordinary sign-in endpoint.
 *
 * That is deliberate: the point of a demo account with a published password is that the demo
 * exercises the real path, and if the real path is broken the demo should show it rather than
 * route around it. The private minting endpoint stays as the fallback so a button can never
 * dead-end on a deployment where sign-in itself is misconfigured. */
async function demoSignIn(button){
  const message=$('authMessage');
  const role=button.dataset.role;
  button.disabled=true;message.textContent='';message.style.color='';
  try{
    const methods=demoMethods||await fetch('/api/auth/methods').then(r=>r.json());
    const account=(methods.demo_accounts||[]).find(a=>a.role===role);
    // Both values come from the API, never from this file: a page that carries the addresses
    // would still show a demo that the server has switched off.
    const email=account&&account.email;
    const password=methods.demo_password;
    let result=null;
    if(email&&password){
      const form=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password})});
      result=await form.json();
      if(!form.ok)result=null;             // fall through to minting rather than dead-ending
    }
    if(!result){
      const minted=await fetch('/api/auth/demo-login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role})});
      result=await minted.json();
      if(!minted.ok)throw Error(result.error||'Demo sign-in failed');
    }
    setToken(result.access_token);
    try{localStorage.setItem(DEMO_ROLE_KEY,role)}catch{}
    $('authModal').hidden=true;
    await loadAccount();
  }catch(error){
    message.textContent=error.message;
    message.style.color='#fda4af';
  }finally{button.disabled=false}
}

/* Put back a demo session that was lost.
 *
 * Sessions here are lost for reasons nobody did wrong: the development database is rebuilt,
 * so the account a token names is recreated; and a hosted preview is served from a hostname
 * that changes when the sandbox does, which moves the whole origin and takes localStorage with
 * it. Both look exactly like being logged out, and both are recoverable when the demo is on —
 * which is already a flag that refuses to run over https, and already means "anyone who can
 * reach this can sign in". So a lost demo session is restored instead of reported, to the same
 * role as last time, and only once per page load so a genuine failure cannot loop.
 */
async function restoreDemoSession(){
  if(restoreAttempted)return false;
  restoreAttempted=true;
  try{
    const methods=demoMethods||await fetch('/api/auth/methods').then(r=>r.json());
    demoMethods=methods;
    if(!methods.demo_login)return false;
    let role='user';
    try{role=localStorage.getItem(DEMO_ROLE_KEY)||'user'}catch{}
    const response=await fetch('/api/auth/demo-login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role})});
    if(!response.ok)return false;
    const result=await response.json();
    setToken(result.access_token);
    return true;
  }catch{return false}
}

/* ------------------------------------------------------------------- account */

function tokenExpiry(){
  try{
    const payload=JSON.parse(atob(accessToken.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
    return payload.exp?new Date(payload.exp*1000):null;
  }catch{return null}
}

async function loadAccount(){
  let user=await sessionUser();
  if(user===undefined){
    // The server did not answer. That is not a session ending, and saying so is the whole
    // point: a flaky moment must not look like being logged out.
    status('The server did not answer just now. Your session is untouched — reload when you like.',true);
    return;
  }
  if(user===null){
    /* The server does not see a session. Two unlike things look identical from here: the
     * credential really is finished, or it never arrived — this page has been seen holding a
     * token and still being answered as a guest, which is a credential lost on the way out
     * rather than at the door. While the demo is on, a session can be put back, so try once
     * before ending anything. Sending somebody to a sign-in card they have just used, without
     * asking whether it was needed, is the whole complaint.
     */
    if(await restoreDemoSession()) user=await sessionUser();
    if(user===undefined){
      status('The server did not answer just now. Your session is untouched — reload when you like.',true);
      return;
    }
    if(user===null){
      // `ended` only if there was something to end: a visitor who never signed in should see
      // the sign-in card, not a notice about a session they did not have.
      endSession('the server does not recognise this browser as signed in',Boolean(accessToken));
      return;
    }
  }
  me=user;
  try{
    $('chipEmail').textContent=me.email;
    $('accountEmail').textContent=me.email;
    $('accountVerified').textContent=me.email_verified?'Yes':'Not yet — check your inbox';
    $('accountRole').textContent=me.role==='admin'?'Administrator':'Member';
    $('resendBtn').hidden=Boolean(me.email_verified);
    $('adminLink').hidden=me.role!=='admin';
    const expiry=tokenExpiry();
    $('sessionExpiry').textContent=expiry?expiry.toLocaleString():'unknown';
  }catch(error){
    // Something other than the session failed: the billing payload, the network, or this
    // page's own rendering. The session stays — clearing it here is what turned a transient
    // error into a logout.
    status('Your account did not finish loading ('+error.message+'). Retrying is safe.',true);
    return;
  }
  try{await loadBilling()}
  catch(error){status('Your balance could not be loaded ('+error.message+').',true)}
}

/* ------------------------------------------------------------------- billing
 * One place for money. The balance, the plan, the packs and the ledger are rendered here
 * and nowhere else — the studio no longer carries a second copy of any of it.
 */
async function loadBilling(){
  const billing=await apiFetch('/api/billing/account').then(r=>r.json());
  renderBilling(billing);
  return billing;
}

function renderBilling(billing){
  const wallet=billing.wallet||{},total=wallet.total||0;
  $('creditBalance').textContent=creditCount(total);
  $('walletTotal').textContent=creditCount(total);
  $('walletPlanCredits').textContent=creditCount(wallet.subscription_credits);
  $('walletTopupCredits').textContent=creditCount(wallet.topup_credits);
  $('walletUsedCredits').textContent=creditCount(billing.credits_used);

  const expiry=$('walletExpiry');
  if(wallet.topup_credits>0&&wallet.topup_expires_at){
    expiry.hidden=false;
    expiry.textContent='Top-up credits are usable until '+walletDate(wallet.topup_expires_at)+'.';
  }else expiry.hidden=true;

  const plan=billing.plan||{};
  $('planName').textContent=plan.name||'Eidomira Live Pro';
  if(plan.price_ngn)$('planPriceNgn').textContent='\u20a6'+creditCount(plan.price_ngn);
  if(plan.price_usd)$('planPriceUsd').textContent='$'+plan.price_usd;
  const features=$('planFeatures');features.replaceChildren();
  (plan.features||[]).forEach(feature=>{const li=document.createElement('li');li.textContent=feature;features.append(li)});

  const sub=billing.subscription,pill=$('accountPlan'),planNote=$('planNote');
  if(sub&&sub.status==='trialing'){
    pill.textContent='TRIAL';pill.className='pill';
    planNote.textContent='Trial ends '+walletDate(sub.current_period_end)+'. Your trial credits stay yours when you upgrade.';
  }else if(sub&&sub.status==='active'){
    pill.textContent=String(sub.plan||'live-pro').toUpperCase();pill.className='pill pill--live';
    planNote.textContent='Active until '+walletDate(sub.current_period_end)+'.';
  }else{
    pill.textContent='NO PLAN';pill.className='pill';
    planNote.textContent='No active plan. Top-up credits work without one; plan credits need a plan.';
  }

  /* The packs are always shown. They used to be hidden outright when the provider was not
   * configured, which meant a deployment with no Paystack key had no payment UI at all and
   * looked like a missing feature rather than a missing setting. Disabled and explained
   * beats invisible. */
  const configured=Boolean(billing.paystack_configured);
  const verified=Boolean(me&&me.email_verified);
  const buyable=configured&&verified;
  const packs=$('walletPacks');packs.replaceChildren();
  (billing.topups||[]).forEach(pack=>{
    const card=document.createElement('button');
    card.type='button';card.className='packCard';card.dataset.product=pack.product;card.disabled=!buyable;
    const credits=document.createElement('b');credits.textContent='+'+creditCount(pack.credits)+' credits';
    const price=document.createElement('span');price.textContent=pack.price;
    const action=document.createElement('small');
    action.textContent=buyable?'Pay with Paystack':(configured?'Verify your email first':'Card payments are off');
    card.append(credits,price,action);
    if(buyable)card.onclick=()=>startCheckout(pack.product);
    packs.append(card);
  });

  const paymentState=$('paystackState'),paymentNote=$('paystackNote');
  if(configured){
    paymentState.textContent='Paystack ready';paymentState.className='badge live';
    paymentNote.textContent='Paystack takes the card on its own page. You come back here when it is done, and the credits are added automatically.';
  }else{
    paymentState.textContent='Paystack not configured';paymentState.className='badge off';
    paymentNote.textContent='This deployment has no Paystack secret key, so no card can be entered and nothing can be charged. Set STUDIO_PAYSTACK_SECRET_KEY and register the webhook to switch these on — the prices below are what it will charge.';
  }

  const state=$('walletState'),message=$('walletMessage');
  if(!configured){state.textContent='Payments off';state.className='badge off'}
  else if(!verified){state.textContent='Verify email';state.className='badge off'}
  else if(total<=0){state.textContent='Empty';state.className='badge off'}
  else{state.textContent='Ready';state.className='badge live'}
  if(!message.textContent&&!configured)
    message.textContent='No credits can be bought on this deployment yet. The buttons above are switched off, not broken.';

  const list=$('walletLedger');list.replaceChildren();
  const entries=billing.ledger||[];
  if(!entries.length){
    const li=document.createElement('li');li.className='note';li.textContent='No credit movements yet.';
    list.append(li);
  }
  entries.forEach(entry=>{
    const li=document.createElement('li');
    const delta=document.createElement('b');
    delta.className=entry.delta>0?'good':'spent';
    delta.textContent=(entry.delta>0?'+':'')+creditCount(entry.delta);
    const what=document.createElement('span');
    what.textContent=(entry.description||entry.kind)+' · '+walletDate(entry.created_at);
    const after=document.createElement('small');
    after.textContent=creditCount(entry.balance_after)+' after';
    li.append(delta,what,after);
    list.append(li);
  });
}

async function startCheckout(product){
  if(!accessToken)return openAuth();
  const message=$('walletMessage');
  note(message,'Opening Paystack…');
  try{
    const response=await apiFetch('/api/payments/paystack/checkout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product})});
    const result=await response.json();
    if(!response.ok)throw Error(result.error||'Checkout is unavailable');
    note(message,'Taking you to Paystack…');
    location.href=result.authorization_url;
  }catch(error){
    showView('billing');
    note(message,error.message,true);
  }
}

/* ------------------------------------------------------------------ password */

$('passwordForm').onsubmit=async event=>{
  event.preventDefault();
  const message=$('pwMessage'),current=$('pwCurrent').value,next=$('pwNew').value,again=$('pwConfirm').value;
  if(next!==again)return note(message,'The two new passwords do not match.',true);
  $('pwSubmit').disabled=true;note(message,'Updating…');
  try{
    const response=await apiFetch('/api/auth/password',{method:'POST',reauth:false,headers:{'Content-Type':'application/json'},body:JSON.stringify({current_password:current,new_password:next})});
    const result=await response.json();
    if(!response.ok)throw Error(result.error||'The password could not be changed');
    $('passwordForm').reset();
    note(message,'Password updated. Other devices stay signed in until their session expires.');
  }catch(error){
    note(message,error.message,true);
  }finally{$('pwSubmit').disabled=false}
};

/* --------------------------------------------------------------- housekeeping */

$('logoutBtn').onclick=async()=>{
  try{await apiFetch('/api/auth/logout',{method:'POST'})}catch{}
  setToken('');
  location.href='/';
};

$('forgetBtn').onclick=async()=>{
  const message=$('forgetMessage');
  note(message,'Signing out…');
  try{await apiFetch('/api/auth/logout',{method:'POST'})}catch{}
  setToken('');
  try{localStorage.removeItem(PREFS_KEY)}catch{}
  note(message,'Signed out. This browser keeps nothing about you now.');
  setTimeout(()=>{location.href='/'},700);
};

$('resendBtn').onclick=async()=>{
  const button=$('resendBtn');button.disabled=true;
  try{
    await apiFetch('/api/auth/resend-verification',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:me?me.email:''})});
    button.textContent='Verification email sent';
  }catch{button.textContent='Could not send — try again'}
  finally{button.disabled=false}
};

$('refreshDevices').onclick=listDevices;
$('refreshDiag').onclick=()=>health();

$('cameraDevice').onchange=event=>{writePrefs({cameraId:event.target.value});note($('prefsMessage'),'Camera preference saved for this browser.')};
$('micDevice').onchange=event=>{writePrefs({micId:event.target.value});note($('prefsMessage'),'Microphone preference saved for this browser.')};
$('motionToggle').onchange=event=>{writePrefs({reduceMotion:event.target.checked});note($('prefsMessage'),event.target.checked?'Motion effects reduced.':'Motion effects restored.')};
document.querySelectorAll('input[name=quality]').forEach(radio=>{
  radio.onchange=()=>{writePrefs({quality:radio.value});note($('prefsMessage'),'Quality preset saved; it applies to your next enrolment.')};
});

/* ---------------------------------------------------------------- boot */

loadAccount();loadAuthMethods();listDevices();applyPrefs();
showView((location.hash||'#studio').slice(1));
if(navigator.mediaDevices?.addEventListener)navigator.mediaDevices.addEventListener('devicechange',listDevices);
const landing=new URLSearchParams(location.search);
if(landing.get('signin')){
  const ended=Boolean(landing.get('ended'));
  history.replaceState({},'',location.pathname);
  openAuth('login');
  if(ended)$('authMessage').textContent='Your session ended, so you have been signed out. Sign in again — this is not an error in your account.';
}
let media=null,pc=null,session=null,running=false,statsTimer=null,reconnects=0;
let recorder=null,recordedChunks=[],recordUrl=null,recordedBlob=null,recordStarted=0,recordTimer=null;
let cameraFacing='user',micStream=null,wakeLock=null,deferredInstall=null;
let callRoomInstance=null,callVideoTrack=null,callAudioTrack=null;

async function health(){
  try{const j=await apiFetch('/api/health').then(r=>r.json());
    $('engineName').textContent=j.backend.toUpperCase();
    $('healthText').textContent=j.gpu?((j.provider?j.provider.toUpperCase():'GPU')+((j.accelerated===false)?' · CPU':' neural engine ready')):'Diagnostic transport mode';
    $('healthDot').style.background=(j.gpu&&j.accelerated!==false)?'#6ee7a5':(j.gpu?'#52d3ff':'#f59e0b');
    reportDiagnostics(j);
  }catch{
    $('healthText').textContent='Engine unavailable';
    reportDiagnostics(null);
  }
}

/* The diagnostics card, written from the same payload that drives the engine chip. A
 * deployment without swap weights reports `diagnostic` here, and that is stated in words
 * rather than left for somebody to infer from a label. */
function reportDiagnostics(j){
  const say=(id,value)=>{$(id).textContent=value};
  if(!j){
    ['diagBackend','diagProvider','diagAccelerated','diagPeers','diagTurn','diagCalls','diagSelfVerify']
      .forEach(id=>say(id,'\u2014'));
    $('diagNote').textContent='The engine could not be reached, so nothing here can be reported.';
    return;
  }
  say('diagBackend',j.backend);
  say('diagProvider',j.provider||'none \u2014 CPU path');
  say('diagAccelerated',j.accelerated===true?'yes':(j.accelerated===false?'no':'unknown'));
  say('diagPeers',(j.active_peers||0)+' of '+(j.peer_capacity||0));
  say('diagTurn',j.turn_configured?'configured':'not configured');
  say('diagCalls',j.calls_configured?'configured':'not configured');
  say('diagSelfVerify',j.self_verification?'required':'not required');
  $('diagNote').textContent=j.backend==='diagnostic'
    ?'This deployment runs the diagnostic backend: no licensed swap weights are installed, so frames are not being face-swapped. Everything else \u2014 enrolment, liveness, transport, recording, credits \u2014 is real and runs against this engine.'
    :'Frames are being processed by '+j.backend+' using the '+((j.provider||'CPU').toUpperCase())+' provider.';
}

function recordedAudioConstraints(){
  const audio={echoCancellation:true,noiseSuppression:true};
  const chosen=readPrefs().micId;
  if(chosen)audio.deviceId={exact:chosen};
  return audio;
}

/* Email-verification links and the return from Paystack are handled by URL, not by a view:
 * the link arrives with ?verify=, the payment with ?payment=return&reference=. Each lands
 * on the page that owns it, which is Settings and Billing respectively. */
async function verifyFromLink(){
  const token=new URLSearchParams(location.search).get('verify');
  if(!token)return;
  history.replaceState({},'',location.pathname);
  showView('studio');
  try{
    const r=await fetch('/api/auth/verify-email',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
    const j=await r.json();
    if(!r.ok)throw Error(j.error||'This verification link is not valid.');
    setToken(j.access_token);
    await loadAccount();
    status('Email verified. Your 7-day trial is active.');
  }catch(error){
    showView('settings');
    note($('prefsMessage'),error.message,true);
  }
}

async function verifyPaymentReturn(){
  const params=new URLSearchParams(location.search);
  const reference=params.get('reference');
  if(params.get('payment')!=='return'||!reference)return;
  history.replaceState({},'',location.pathname+'#billing');
  showView('billing');
  if(!accessToken)return openAuth('login');
  note($('walletMessage'),'Confirming the Paystack payment\u2026');
  try{
    const r=await apiFetch('/api/payments/paystack/verify/'+encodeURIComponent(reference));
    const j=await r.json();
    if(!r.ok)throw Error(j.error||'The payment could not be verified.');
    if(j.status==='success'){
      note($('walletMessage'),j.kind==='topup'?'Payment confirmed. Your credits have been added.':'Payment confirmed. Live Pro is active.');
      await loadBilling();
    }else note($('walletMessage'),'Paystack reports this payment as '+j.status+'. Nothing has been charged.',true);
  }catch(error){
    note($('walletMessage'),error.message,true);
  }
}
health();

let selectedPresetBlob=null;
document.querySelectorAll('.presetBtn').forEach(btn=>{
  btn.onclick=async()=>{
    try{
      const src=btn.dataset.src;
      status('Loading specimen '+btn.textContent.trim()+'…');
      const res=await fetch(src);
      selectedPresetBlob=await res.blob();
      $('sourcePreview').src=src;
      $('uploadCard').classList.add('hasImage');
      $('consent').checked=true;
      status('Selected '+btn.textContent.trim()+'. Click Enroll identity to activate.');
    }catch(e){
      status('Failed to load specimen: '+e.message,true);
    }
  };
});

$('source').onchange=e=>{
  const f=e.target.files[0];
  if(f){
    selectedPresetBlob=null;
    $('sourcePreview').src=URL.createObjectURL(f);
    $('uploadCard').classList.add('hasImage');
  }
};

$('enrollBtn').onclick=async()=>{
  let f=$('source').files[0]||selectedPresetBlob;
  if(!f)return status('Choose a reference portrait or select a preset.',true);
  if(!$('consent').checked)return status('Confirm self-only consent first.',true);
  const d=new FormData();
  d.append('image',f,f.name||'identity.jpg');
  d.append('consent','true');
  d.append('quality',readPrefs().quality);
  status('Analyzing identity with neural network…');$('enrollBtn').disabled=true;
  try{
    const r=await apiFetch('/api/sessions',{method:'POST',body:d}),j=await r.json();
    if(!r.ok)throw Error(j.error||'Enrollment failed');
    session=j.session_id;status('Identity enrolled! Face transformation ready.');
    if(j.expires_in)$('privacyTtl').textContent='Ends '+Math.round(j.expires_in/60)+' minutes after enrolling';
    $('p1').classList.add('active');$('p2').classList.add('active');
    if(media)$('goBtn').disabled=false;
    if(running){
      status('Switching to enrolled identity…');
      stopTransform(false);
      setTimeout(startTransform,400);
    }
  }catch(e){status(e.message,true)}finally{$('enrollBtn').disabled=false}
};
function status(t,error=false){$('status').textContent=t;$('status').style.color=error?'#fda4af':'#86efac'}

let syntheticAnimId=null;
function createSyntheticFeed(){
  const canvas=document.createElement('canvas');
  canvas.width=640;canvas.height=480;
  const ctx=canvas.getContext('2d');
  let angle=0;
  function draw(){
    angle+=0.04;
    ctx.fillStyle='#0f111a';ctx.fillRect(0,0,640,480);
    const cx=320+Math.sin(angle)*35;
    const cy=240+Math.cos(angle*1.3)*15;
    ctx.fillStyle='#e4be9e';
    ctx.beginPath();ctx.ellipse(cx,cy,95,125,0,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#1c1c24';
    const blink=Math.sin(angle*2.5)>0.94;
    if(blink){
      ctx.fillRect(cx-45,cy-20,26,4);
      ctx.fillRect(cx+19,cy-20,26,4);
    }else{
      ctx.beginPath();ctx.ellipse(cx-32,cy-20,11,7,0,0,Math.PI*2);
      ctx.ellipse(cx+32,cy-20,11,7,0,0,Math.PI*2);ctx.fill();
    }
    ctx.beginPath();ctx.arc(cx,cy+45,24,0.2,Math.PI-0.2);ctx.lineWidth=4;ctx.strokeStyle='#c45a5a';ctx.stroke();
    ctx.fillStyle='#86efac';ctx.font='14px monospace';
    ctx.fillText('● TEST VIDEO FEED (CAMERA BYPASS)',18,30);
    syntheticAnimId=requestAnimationFrame(draw);
  }
  draw();
  return canvas.captureStream?canvas.captureStream(30):null;
}

$('cameraBtn').onclick=async()=>{
  if(media){stopCamera();return}
  try{
    const prefs=readPrefs();
    const video={width:{ideal:1920,max:1920},height:{ideal:1080,max:1080},frameRate:{ideal:30,max:30},facingMode:{ideal:cameraFacing}};
    if(prefs.cameraId)video.deviceId={exact:prefs.cameraId};
    try{
      media=await navigator.mediaDevices.getUserMedia({video,audio:false});
    }catch(error){
      // A saved device that is no longer plugged in must not be a camera that never starts:
      // fall back to the default and say which device was missing.
      if(!prefs.cameraId)throw error;
      delete video.deviceId;
      media=await navigator.mediaDevices.getUserMedia({video,audio:false});
      note($('prefsMessage'),'The saved camera was not available, so the default is in use. Choose another in Settings.',true);
    }
    $('video').srcObject=media;await $('video').play();
    $('cameraEmpty').style.display='none';$('cameraBadge').textContent='LIVE';$('cameraBadge').className='badge live';
    $('cameraBtn').textContent='Stop camera';$('flipBtn').disabled=false;$('p2').classList.add('active');$('goBtn').disabled=!session;
    $('syntheticCamBtn').hidden=true;
  }catch(e){
    status('Camera blocked: '+e.message,true);
    $('syntheticCamBtn').hidden=false;
    $('syntheticCamBtn').scrollIntoView({behavior:'smooth',block:'nearest'});
  }
};
$('syntheticCamBtn').onclick=async()=>{
  if(media){stopCamera();return}
  try{
    media=createSyntheticFeed();
    if(!media)throw Error('Browser does not support canvas video stream');
    $('video').srcObject=media;await $('video').play();
    $('cameraEmpty').style.display='none';$('cameraBadge').textContent='TEST FEED';$('cameraBadge').className='badge live';
    $('cameraBtn').textContent='Stop camera';$('flipBtn').disabled=true;$('p2').classList.add('active');$('goBtn').disabled=!session;
    $('syntheticCamBtn').hidden=true;
    status('Test video feed running. Click Start transformation.');
  }catch(e){status('Test stream error: '+e.message,true)}
};
function stopCamera(){
  if(syntheticAnimId){cancelAnimationFrame(syntheticAnimId);syntheticAnimId=null}
  stopTransform();media?.getTracks().forEach(t=>t.stop());media=null;$('video').srcObject=null;$('cameraEmpty').style.display='grid';$('cameraBadge').textContent='OFFLINE';$('cameraBadge').className='badge off';$('cameraBtn').innerHTML='Start camera <b>→</b>';$('goBtn').disabled=true;$('flipBtn').disabled=true
}
$('flipBtn').onclick=async()=>{const wasRunning=running;if(wasRunning)stopTransform();media?.getTracks().forEach(t=>t.stop());media=null;cameraFacing=cameraFacing==='user'?'environment':'user';await $('cameraBtn').click();status(cameraFacing==='user'?'Front camera selected.':'Rear camera selected.');};

$('goBtn').onclick=()=>running?stopTransform():startTransform();
async function waitForIce(connection){
  if(connection.iceGatheringState==='complete')return;
  await new Promise(resolve=>{const check=()=>{if(connection.iceGatheringState==='complete'){connection.removeEventListener('icegatheringstatechange',check);resolve()}};connection.addEventListener('icegatheringstatechange',check);setTimeout(resolve,4000)});
}
let liveWs=null,wsFrameTimer=null,wsCapCanvas=null,wsOutCanvas=null;

function startWebSocketTransform(){
  if(pc){try{pc.close()}catch(_){}pc=null}
  status('Live neural stream connecting…');
  const protocol=location.protocol==='https:'?'wss:':'ws:';
  const url=`${protocol}//${location.host}/api/live/${session}`;
  if(!wsCapCanvas)wsCapCanvas=document.createElement('canvas');
  if(!wsOutCanvas)wsOutCanvas=document.createElement('canvas');
  const capCtx=wsCapCanvas.getContext('2d'),outCtx=wsOutCanvas.getContext('2d');
  try{liveWs=new WebSocket(url);liveWs.binaryType='arraybuffer'}catch(err){status('Stream error: '+err.message,true);stopTransform();return}
  let inFlight=false;
  liveWs.onopen=()=>{
    running=true;$('goBtn').disabled=false;$('goBtn').innerHTML='<span>■</span> Stop transformation';
    $('outputBadge').textContent='LIVE';$('outputBadge').className='badge live';$('p3').classList.add('active');
    status('Live neural stream connected.');
    if(wsOutCanvas.captureStream){
      const stream=wsOutCanvas.captureStream(30);
      $('output').srcObject=stream;$('output').style.display='block';$('outputEmpty').style.display='none';
      $('recordBtn').disabled=false;$('cleanBtn').disabled=false;$('output').play().catch(()=>{});
    }
    wsFrameTimer=setInterval(()=>{
      if(!running||inFlight||!liveWs||liveWs.readyState!==WebSocket.OPEN)return;
      const vid=$('video');if(!vid||vid.videoWidth===0)return;
      wsCapCanvas.width=vid.videoWidth||640;wsCapCanvas.height=vid.videoHeight||480;
      capCtx.drawImage(vid,0,0,wsCapCanvas.width,wsCapCanvas.height);
      inFlight=true;
      wsCapCanvas.toBlob(blob=>{
        if(blob&&liveWs&&liveWs.readyState===WebSocket.OPEN){
          blob.arrayBuffer().then(buf=>liveWs.send(buf)).catch(()=>{inFlight=false});
        }else{inFlight=false}
      },'image/jpeg',0.82);
    },45);
  };
  liveWs.onmessage=async(e)=>{
    if(typeof e.data==='string'){
      try{
        const msg=JSON.parse(e.data);
        if(msg.type==='frame'){
          $('latency').textContent=msg.latency_ms+' ms';
          $('frameCost').textContent=(msg.server_ms!=null?msg.server_ms+' ms':'— ms');
          $('outputBadge').textContent=msg.face_found?'LIVE':'NO FACE';
          $('inferenceSize').textContent=(wsCapCanvas?wsCapCanvas.width:640)+' px';
          if(msg.server_ms)$('fps').textContent=Math.round(1000/Math.max(msg.server_ms,35))+' FPS';
        }else if(msg.type==='verification'){
          $('verified').textContent=msg.verified?'VERIFIED':'MATCHING';
        }else if(msg.type==='fatal'){
          status(msg.message,true);stopTransform();
        }
      }catch(_){}
    }else{
      inFlight=false;
      const blob=new Blob([e.data],{type:'image/jpeg'});
      const img=new Image();
      img.onload=()=>{
        wsOutCanvas.width=img.naturalWidth||640;wsOutCanvas.height=img.naturalHeight||480;
        outCtx.drawImage(img,0,0,wsOutCanvas.width,wsOutCanvas.height);
        URL.revokeObjectURL(img.src);
      };
      img.src=URL.createObjectURL(blob);
    }
  };
  liveWs.onerror=()=>{inFlight=false};
  liveWs.onclose=()=>{
    inFlight=false;clearInterval(wsFrameTimer);wsFrameTimer=null;
    if(running)stopTransform();
  };
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

    // If WebRTC UDP doesn't reach connected in 2.5s (e.g. cloud NAT without UDP), switch to WebSocket stream
    const fallbackTimer=setTimeout(()=>{
      if(running&&(!pc||pc.connectionState!=='connected')){
        startWebSocketTransform();
      }
    },2500);

    pc.onconnectionstatechange=()=>{
      const state=pc?.connectionState;
      if(state==='connected'){clearTimeout(fallbackTimer);status('Live neural channel connected.');reconnects=0}
      if(['failed','disconnected'].includes(state)&&running){clearTimeout(fallbackTimer);startWebSocketTransform()}
    };
    const offer=await pc.createOffer();await pc.setLocalDescription(offer);await waitForIce(pc);
    const response=await apiFetch('/api/webrtc/'+session+'/offer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
    const answer=await response.json();if(!response.ok)throw Error(answer.error||'WebRTC negotiation failed');
    await pc.setRemoteDescription(answer);
    try{wakeLock=await navigator.wakeLock?.request('screen')}catch{}
    running=true;$('goBtn').disabled=false;$('goBtn').innerHTML='<span>■</span> Stop transformation';
    $('outputBadge').textContent='CONNECTING';$('outputBadge').className='badge live';$('p3').classList.add('active');
    statsTimer=setInterval(updateWebRTCStats,1000);
  }catch(e){
    // If WebRTC fails outright (e.g. browser ICE block), fall back to WebSocket immediately
    startWebSocketTransform();
  }
}
function handleTelemetry(event){
  const j=JSON.parse(event.data);
  if(j.type==='liveness'){$('livenessPrompt').style.display=j.complete?'none':'block';$('livenessText').textContent=j.instruction;$('livenessBar').style.width=Math.round(j.progress*100)+'%';$('verified').textContent=j.complete?'MATCHING':'LIVE CHECK';if(j.expired)status(j.instruction,true)}
  if(j.type==='verification'){$('livenessPrompt').style.display='none';$('verified').textContent=j.verified?'VERIFIED':'FAILED';if(!j.verified)status('Live face does not match the enrolled identity.',true)}
  if(j.type==='metrics'){$('latency').textContent=j.inference_ms+' ms';$('frameCost').textContent=(j.frame_ms!=null?j.frame_ms+' ms':'— ms');$('fps').textContent=j.fps+' FPS';$('dropped').textContent=j.dropped;$('inferenceSize').textContent=j.inference_width+' px';$('outputBadge').textContent=j.face_found?'LIVE':'NO FACE'}
  if(j.type==='trainer')renderTrainer(j);
  if(j.type==='error')status(j.message,true);
}
function renderTrainer(state){
  const badge=$('trainerState');
  badge.textContent=!state.monitoring?'Off':(state.pending?'Tuning '+state.pending:'Watching');
  badge.className=!state.monitoring?'off':(state.pending?'tuning':'watching');
  const parts=[];
  if(state.samples)parts.push(state.samples+(state.samples===1?' frame measured':' frames measured'));
  parts.push(state.mistakes?(state.mistakes+(state.mistakes===1?' defect':' defects')):'no defects found');
  if(state.kept)parts.push(state.kept+(state.kept===1?' setting improved':' settings improved'));
  if(state.reverted)parts.push(state.reverted+' reverted');
  if(state.repairs&&state.repairs.length)parts.push(state.repairs.length+(state.repairs.length===1?' stage repaired':' stages repaired'));
  $('trainerSummary').textContent=state.monitoring?parts.join(' · '):'Not monitoring this session.';
  const rows=[];
  (state.repairs||[]).forEach(r=>rows.push(['good',r.stage+': '+r.action]));
  (state.recent||[]).forEach(m=>rows.push([m.severity==='critical'?'bad':(m.severity==='warning'?'warn':'note'),m.summary]));
  (state.notes||[]).forEach(n=>rows.push(['note',n]));
  $('trainerList').replaceChildren(...rows.slice(-4).map(([kind,text])=>{
    const li=document.createElement('li');li.className=kind;li.textContent=text;return li;
  }));
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
  if(wsFrameTimer){clearInterval(wsFrameTimer);wsFrameTimer=null}
  if(liveWs){try{liveWs.close()}catch(_){}liveWs=null}
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
  try{const tracks=[...outputStream.getVideoTracks()];if($('recordMic').checked){micStream=await navigator.mediaDevices.getUserMedia({audio:recordedAudioConstraints(),video:false});tracks.push(...micStream.getAudioTracks())}const stream=new MediaStream(tracks);const mime=preferredMime();const options={videoBitsPerSecond:6_000_000,audioBitsPerSecond:128_000};if(mime)options.mimeType=mime;recorder=new MediaRecorder(stream,options);
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

verifyFromLink();verifyPaymentReturn();
