var jsdom = require("jsdom");
var read = require('node-readability');
var XMLHttpRequest = require('xhr2');

jsdom.env({
  url: "http://online.wsj.com/public/page/archive-2010-1-15.html",
  scripts: ["http://code.jquery.com/jquery.js"],
  done: function (errors, window) {
    var $ = window.$;
    //console.log("HN Links");
    console.log($("div#archivedArticles").find("a").length);
    $("div#archivedArticles").find("a").each(function() {
      console.log(" -", $(this).attr('href'));

    //read($(this).attr('href'), function(err, article, meta) {
  // Main Article
  //console.log(article.content);
  // Title
  //console.log(article.title);

  // HTML Source Code
  //console.log(article.html);
  // DOM
  //console.log(article.document);

  // Response Object from Request Lib
  //console.log(meta);
 // });
    });
  }
});


var xmlhttp;
xmlhttp = new XMLHttpRequest();
xmlhttp.onreadystatechange = function() {
    if (XMLHttpRequest.DONE == xmlhttp.readyState) {
        if(200 == xmlhttp.status && xmlhttp.readyState==4) {
          console.log(xmlhttp.responseText);
        }
        else if (400 == xmlhttp.status) {
            console.log("failed");
        }
        else {
            console.log(xmlhttp.response);
            console.log("something else other than 200 was returned");
        }
    }
}

