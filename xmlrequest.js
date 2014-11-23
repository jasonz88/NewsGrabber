var XMLHttpRequest = require('xhr2');
var xmlhttp;
xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (XMLHttpRequest.DONE == xmlhttp.readyState) {
        if(200 == xmlhttp.status && xmlhttp.readyState==4) {
          var vv = JSON.parse(xmlhttp.responseText);
            console.log(vv.response.docs[3].snippet);
        }
        else if (400 == xmlhttp.status) {
            console.log("feedback submit failed");
        }
        else {
            console.log(xmlhttp.response);
            console.log("feedback submit: something else other than 200 was returned");
        }
    }
}

xmlhttp.open("GET", "http://api.nytimes.com/svc/search/v2/articlesearch.json?q=new+york+times&page=2&sort=oldest&api-key=e99277b69a4b7abebb9153d3cb535037:5:70221482", true);
xmlhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
xmlhttp.send(null);    
