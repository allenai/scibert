#!/usr/bin/python3

import xml.sax
import collections
import os

class PubHandler(xml.sax.ContentHandler):
   def __init__(self):
       # content of each publication document
        self.id = ""
        self.journalname = ""  # journal name
        self.openaccess = ""   # open access or not
        self.pubdate = ""      # publication date date
        self.title = ""        # title
        self.authors = []      # list of authors
        self.keyphrases = []   # author-defined keyphrases
        self.abstract = ""     # abstract
        self.highlights = []   # highlight (i.e. author-defined summary statements)
        self.text = collections.OrderedDict()  # text
        self.captions = []     # image and table captions
        self.bib_entries = []  # bib entries

        # temporary variables, ignore
        self.CurrentData = ""
        self.highlightflag = False
        self.textbuilder_highlight = []
        self.inpara = False
        self.textbuilder = []
        self.textbuilder_abstract = []
        self.paraid = 0
        self.inabstract = False
        self.textbuilder_title = []
        self.intitle = False
        self.textbuilder_captions = []
        self.incaption = False
        self.textbuilder_bib = []
        self.inbib = False

   def startElement(self, tag, attributes):
       # Call when an element starts
       self.CurrentData = tag
       v = attributes.get("class")  # returns "None" if not contained
       if v == "author-highlights":
           self.highlightflag = True
       if tag == "ce:para":
           self.inpara = True
           self.paraid += 1  # attributes.get("id")  # there isn't always one, so let's use a counter instead
       if tag == "ce:abstract" and self.highlightflag == False:
           self.inabstract = True
       if tag == "ce:title" or tag == "dc:title":
           self.intitle = True
       if tag == "ce:caption":
           self.incaption = True
       if tag == "ce:bib-reference":
           self.inbib = True

   def endElement(self, tag):
       # Call when an elements ends
       self.CurrentData = ""
       if tag == "ce:abstract":
           self.highlightflag = False
           self.inabstract = False
           if len(self.textbuilder_abstract) > 0:
               para = "".join(self.textbuilder_abstract)
               self.abstract = para
               self.textbuilder_abstract = []
       elif tag == "ce:para":
           if len(self.textbuilder_highlight) > 0:
               para = "".join(self.textbuilder_highlight)
               self.highlights.append(para)
               self.textbuilder_highlight = []
           self.inpara = False
           if len(self.textbuilder) > 0:
               para = "".join(self.textbuilder)
               self.text[self.paraid] = para
               self.textbuilder = []
       if tag == "ce:title" or tag == "dc:title":
           self.intitle = False
           if len(self.textbuilder_title) > 0:
               para = "".join(self.textbuilder_title)
               self.title = para
               self.textbuilder_title = []
       if tag == "ce:caption":
           if len(self.textbuilder_captions) > 0:
               caption = " ".join(self.textbuilder_captions)
               self.captions.append(caption)
               self.textbuilder_captions = []
       if tag == "ce:bib-reference":
           if len(self.textbuilder_bib) > 0:
               bibref = " ".join(self.textbuilder_bib)
               self.bib_entries.append(bibref)
               self.textbuilder_bib = []

   def characters(self, content):
       # Call when a character is read
       if self.CurrentData == "dc:identifier":
           self.id = content
       elif self.CurrentData == "prism:publicationName":
           self.journalname = content
       elif self.CurrentData == "openaccess":
           self.openaccess = content
       elif self.CurrentData == "prism:coverDate":
           self.pubdate = content
       elif self.intitle == True:
           self.textbuilder_title.append(content)
       elif self.CurrentData == "dc:creator":
           self.authors.append(content)
       elif self.CurrentData == "dcterms:subject":
           self.keyphrases.append(content)
       elif (self.CurrentData == "dc:description") or (self.inabstract == True and self.highlightflag == False):
           if content.startswith("Abstract"):
               content = content.replace("Abstract", "", 1)
           self.textbuilder_abstract.append(content)
       elif self.highlightflag == True:
           if content.startswith("Highlights"):
               content = content.replace("Highlights", "", 1)
           if content.startswith("•"):
               content = content.replace("•", "", 1)
           self.textbuilder_highlight.append(content)
       elif self.inpara == True and self.highlightflag == False:
           self.textbuilder.append(content)
       elif self.incaption == True:
           self.textbuilder_captions.append(content)
       elif self.inbib == True:
           self.textbuilder_bib.append(content)


def parseXML(fpath="papers_with_highlights/S2352220815001534.xml"):
    '''
    Parse XML files to retrieve full publication text
    :param fpath: path to file
    :return:
    '''

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = PubHandler()
    parser.setContentHandler(Handler)

    # parse document
    parser.parse(fpath)

    # access the different parts of the publication
    print("Title:", Handler.title)
    for h in Handler.highlights:
        print("Highlight:", h)
    print("Abstract:", Handler.abstract)
    for n, t in Handler.text.items():
        print("Text:", t)


def parseXMLAll(dirpath = "papers_with_highlights/"):

    dir = os.listdir(dirpath)
    for f in dir:
        if not f.endswith(".xml"):
            continue
        print(f)
        parseXML(os.path.join(dirpath, f))
        print("")


if __name__ == '__main__':
    parseXML()
