var read = require('node-readability');

var content = '';
process.stdin.resume();
process.stdin.on('data', function(buf) { content += buf.toString(); });
process.stdin.on('end', function() {
    read(content, function(err, article, meta) {
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

  // Close article to clean up jsdom and prevent leaks
  //article.close();
});
});



