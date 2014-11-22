// Count all of the links from the Node.js build page
/*
var jsdom = require("jsdom");

jsdom.env(
  "http://query.nytimes.com/search/sitesearch/#/crude+oil/from20100502to20100602/allresults/1/allauthors/relevance/business",
  ["http://code.jquery.com/jquery.js"],
  function (errors, window) {
    console.log("there have been", window.$("a").length, "nodejs releases!");
  }
);
*/
// Print all of the news items on Hacker News
var jsdom = require("jsdom");
var read = require('node-readability');

jsdom.env({
  url: "http://online.wsj.com/search/search_center.html",
  scripts: ["http://code.jquery.com/jquery.js"],
  done: function (errors, window) {
    var $ = window.$;
    console.log("HN Links");
    $("a.pb12").each(function() {
      console.log(" -", $(this).attr('href'));
    read($(this).attr('href'), function(err, article, meta) {
  // Main Article
  console.log(article.content);
  // Title
  //console.log(article.title);

  // HTML Source Code
  //console.log(article.html);
  // DOM
  //console.log(article.document);

  // Response Object from Request Lib
  //console.log(meta);
});
    });
  }
});


/*
var XMLHttpRequest = require('xhr2');
var xmlhttp;
xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (XMLHttpRequest.DONE == xmlhttp.readyState) {
        if(200 == xmlhttp.status && xmlhttp.readyState==4) {
            var div = window.content.document.createElement('div');
            div.innerHTML = xmlhttp.responseText;
            var elements = div.getElementsById('searchResults');
            console.log(elements);
        }
        else if (400 == xmlhttp.status) {
            console.log("feedback submit failed");
        }
        else {
            console.log("feedback submit: something else other than 200 was returned");
        }
    }
}

xmlhttp.open("GET", "http://query.nytimes.com/search/sitesearch/#/crude+oil/from20100502to20100602/allresults/1/allauthors/relevance/business", true);
xmlhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
xmlhttp.send(null);    


var request = require('request'),
        dom = require('node-dom').dom,
        fs = require('fs'),   
        URL = require('url');

    var    args = require('tav').set({
                    url:{
                    note:'URL of the page to parse'
                    }
                },'node-dom for node.js',true);

    var url = URL.parse(args.url);

    var req = {uri:url.href};

    request(req,function (error, response, page) {

        if (!error && response.statusCode == 200) {

            var options =    {    url:url,
                                features: {
                                            FetchExternalResources  : {script:'', img:'', input:'', link:''},
                                            ProcessExternalResources: {script:'',img:'',link:'',input:''},
                                            removeScript: true //Remove scripts for innerHTML and outerHTML output
                                }
            };

            window=dom(page,null,options); //global

            document=window.document; //global

            document.onload=function() {
            //Warning : you are not in the window context here (ie you can not access window's global var as global variables directly)
            //Context are explained here https://github.com/joyent/node/issues/1674

                fs.writeFile('./outer.html', document.html.outerHTML, function (err) {});
                //check the result in outer.html file
                //to test the result in a browser, don't forget to put the base tag after <head> with the correct href
                };
        };
    });

    */