#!/usr/bin/python

import xml.sax
import collections
import os

class PubHandler(xml.sax.ContentHandler):
   def __init__(self):
       # content of each publication document
        self.id = ""
        self.journalname = ""
        self.openaccess = ""
        self.pubdate = ""
        self.title = ""
        self.authors = []
        self.keyphrases = []
        self.abstract = ""
        self.highlights = []
        self.text = collections.OrderedDict()

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


   def startElement(self, tag, attributes):
       # Call when an element starts
        self.CurrentData = tag
        v = attributes.get("class")  # returns "None" if not contained
        if v == "author-highlights":
            self.highlightflag = True
        if tag == "ce:para":
            self.inpara = True
            self.paraid += 1 #attributes.get("id")  # there isn't always one, so let's use a counter instead
        if tag == "ce:abstract" and self.highlightflag == False:
            self.inabstract = True
        if tag == "ce:title" or tag == "dc:title":
            self.intitle = True


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


def parseXML(fpath="data/dev/S0010938X13003818.xml"):
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

    parser.parse(fpath)

    print("Title:", Handler.title)
    for h in Handler.highlights:
        print("Highlight:", h)
    print("Abstract:", Handler.abstract)
    for n, t in Handler.text.items():
        print("Text:", t)


def parseXMLAll(dirpath = "data/dev/"):

    dir = os.listdir(dirpath)
    for f in dir:
        if not f.endswith(".xml"):
            continue
        print(f)
        parseXML(os.path.join(dirpath, f))
        print("")


def readAnn(textfolder = "data/dev/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

    flist = os.listdir(textfolder)
    for f in flist:
        if not f.endswith(".ann"):
            continue
        f_anno = open(os.path.join(textfolder, f), "rU")
        f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        for l in f_anno:
            anno_inst = l.strip("\n").split("\t")
            if len(anno_inst) == 3:
                anno_inst1 = anno_inst[1].split(" ")
                if len(anno_inst1) == 3:
                    keytype, start, end = anno_inst1
                else:
                    keytype, start, _, end = anno_inst1
                if not keytype.endswith("-of"):

                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    if keyphr_text_lookup != keyphr_ann:
                        print("Spans don't match for anno " + l.strip() + " in file " + f)


if __name__ == '__main__':
    #parseXML()
    readAnn()