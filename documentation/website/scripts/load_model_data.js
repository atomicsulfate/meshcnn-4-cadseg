scripts = document.getElementsByClassName('model');

promises = [];
for(var i=0; i < scripts.length; ++i) {
    const script = scripts[i];
    var src = script.getAttribute("src")
    promises.push(fetch(src).then(response => {
       if (!response.ok) {
           throw new Error("HTTP error " + response.status);
       }
       return  response.text();
    }));
}
Promise.all(promises).then((scriptCodes) => {
    for(var i=0; i < scripts.length; ++i) {
    var script = scripts[i];
    var scriptCode = scriptCodes[i];
    //console.log(`Loaded script with size ${scriptCode.length}`);
    script.innerHTML = scriptCode;
    }

    const embedScript = document.createElement('script');
    document.body.appendChild(embedScript);
    //embedScript.async = true;
    embedScript.src = 'https://unpkg.com/@jupyter-widgets/html-manager@^0.18.0/dist/embed-amd.js';
});

