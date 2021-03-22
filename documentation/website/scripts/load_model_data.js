scripts = document.getElementsByClassName('model');

for(var i=0; i < scripts.length; ++i) {
    const script = scripts[i];
    var src = script.getAttribute("src")
     fetch(src).then(response => {
       if (!response.ok) {
           throw new Error("HTTP error " + response.status);
       }
       return  response.text();
    }).then(text => {
           console.log(`Loaded script with size ${text.length}`);
        script.innerHTML = text;
    });
}