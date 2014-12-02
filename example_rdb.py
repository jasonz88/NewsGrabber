from readability.readability import Document
import urllib
html = urllib.urlopen('http://open.blogs.nytimes.com/2009/02/04/announcing-the-article-search-api/?_r=0').read()
readable_article = Document(html).summary()
readable_title = Document(html).short_title()

print readable_article
'''http://query.nytimes.com/search/sitesearch/#/crude+oil/from20100502to201000502/allresults/1/allauthors/relevance/business/'''