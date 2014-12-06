/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtWidgets>
#include <QtNetwork>
#include <QtWebKitWidgets>
#include <QDebug>
#include "mainwindow.h"

/*
 * login nytimes.com and wsj.com
 * *URL generation,   "http://query.nytimes.com/search/sitesearch/#/crude+oil/from20100502to20100503/allresults/1/allauthors/relevance/business"
 * page loading, href capturing div#searchResults
 * readability filter
 * html tags removal
 * *save to file
 */


void HTML2Text(QString html, QString url)
{
//    return in.remove(QRegularExpression("(<!--((.|\n)*?)-->|<[^>]*>|»)"));
    QProcess process;
    process.start("node /home/dyz/MLProj/dataCollect/reada.js");
    process.waitForStarted();
//        qDebug() << view->page()->mainFrame()->toHtml();
//        (<span[^>]*>((.|\n)*?)<\/span>|<\/span>)
//        <span[^>]*>((.|\n)*?)<\/span>

    process.write(html.remove(QRegularExpression("(<span[^>]*>((.|\n)*?)<\/span>|<\/span>)")).toStdString().c_str());
    process.closeWriteChannel();
    process.waitForFinished(-1); // will wait forever until finished

    QString stdout = process.readAllStandardOutput();
    QString stderr = process.readAllStandardError();
    qDebug() << url;
    qDebug() << stdout.remove(QRegularExpression("(<!--((.|\n)*?)-->|<[^>]*>|»|\n)"));;
//        qDebug() << stderr;
}

//! [1]

MainWindow::MainWindow(const QUrl& url)
    :m_isContent(false),
      m_currentURL("")
{
    progress = 0;

    QFile file;
    file.setFileName(":/jquery.min.js");
    file.open(QIODevice::ReadOnly);
    jQuery = file.readAll();
    jQuery.append("\nvar qt = { 'jQuery': jQuery.noConflict(true) };");
    file.close();
//! [1]


    QNetworkProxyFactory::setUseSystemConfiguration(true);

//! [2]
    view = new QWebView(this);
    view->load(QUrl("http://online.wsj.com"));
//    view->load(url);
    connect(view, SIGNAL(loadFinished(bool)), SLOT(adjustLocation()));
    connect(view, SIGNAL(titleChanged(QString)), SLOT(adjustTitle()));
    connect(view, SIGNAL(loadProgress(int)), SLOT(setProgress(int)));
    connect(view->page()->mainFrame(), SIGNAL(loadFinished(bool)), SLOT(finishLoading(bool)));

    connect(view->page()->networkAccessManager(), SIGNAL(sslErrors(QNetworkReply*,QList<QSslError>)),
            SLOT(sslErrors(QNetworkReply*,QList<QSslError>)));

    locationEdit = new QLineEdit(this);
    locationEdit->setSizePolicy(QSizePolicy::Expanding, locationEdit->sizePolicy().verticalPolicy());
    connect(locationEdit, SIGNAL(returnPressed()), SLOT(changeLocation()));

    QToolBar *toolBar = addToolBar(tr("Navigation"));
    toolBar->addAction(view->pageAction(QWebPage::Back));
    toolBar->addAction(view->pageAction(QWebPage::Forward));
    toolBar->addAction(view->pageAction(QWebPage::Reload));
    toolBar->addAction(view->pageAction(QWebPage::Stop));
    toolBar->addWidget(locationEdit);
//! [2]

    QMenu *viewMenu = menuBar()->addMenu(tr("&View"));
    QAction* viewSourceAction = new QAction("Page Source", this);
    connect(viewSourceAction, SIGNAL(triggered()), SLOT(viewSource()));
    viewMenu->addAction(viewSourceAction);

//! [3]
    QMenu *effectMenu = menuBar()->addMenu(tr("&Effect"));
    effectMenu->addAction("Highlight all links", this, SLOT(highlightAllLinks()));

    rotateAction = new QAction(this);
    rotateAction->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    rotateAction->setCheckable(true);
    rotateAction->setText(tr("Turn images upside down"));
    connect(rotateAction, SIGNAL(toggled(bool)), this, SLOT(rotateImages(bool)));
    effectMenu->addAction(rotateAction);

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction(tr("Remove GIF images"), this, SLOT(removeGifImages()));
    toolsMenu->addAction(tr("Remove all inline frames"), this, SLOT(removeInlineFrames()));
    toolsMenu->addAction(tr("Remove all object elements"), this, SLOT(removeObjectElements()));
    toolsMenu->addAction(tr("Remove all embedded elements"), this, SLOT(removeEmbeddedElements()));

    setCentralWidget(view);
    setUnifiedTitleAndToolBarOnMac(true);
}


void MainWindow::sslErrors(QNetworkReply *reply, const QList<QSslError> &error)
{
    // check if SSL certificate has been trusted already
    QString replyHost = reply->url().host() + QString(":%1").arg(reply->url().port());
    reply->ignoreSslErrors();
}


//! [3]

void MainWindow::viewSource()
{
    QNetworkAccessManager* accessManager = view->page()->networkAccessManager();
    QNetworkRequest request(view->url());
    QNetworkReply* reply = accessManager->get(request);
    connect(reply, SIGNAL(finished()), this, SLOT(slotSourceDownloaded()));
}

void MainWindow::slotSourceDownloaded()
{
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(const_cast<QObject*>(sender()));
    QTextEdit* textEdit = new QTextEdit(NULL);
    textEdit->setAttribute(Qt::WA_DeleteOnClose);
    textEdit->show();
    textEdit->setPlainText(reply->readAll());
    reply->deleteLater();
}

//! [4]
void MainWindow::adjustLocation()
{
    locationEdit->setText(view->url().toString());
}

void MainWindow::changeLocation()
{
    QUrl url = QUrl::fromUserInput(locationEdit->text());
    view->load(url);
    view->setFocus();
}
//! [4]

//! [5]
void MainWindow::adjustTitle()
{
    if (progress <= 0 || progress >= 100)
        setWindowTitle(view->title());
    else
        setWindowTitle(QString("%1 (%2%)").arg(view->title()).arg(progress));
}

void MainWindow::setProgress(int p)
{
    progress = p;
    adjustTitle();
}
//! [5]

//! [6]
void MainWindow::finishLoading(bool)
{
    progress = 100;
    adjustTitle();
    view->page()->mainFrame()->evaluateJavaScript(jQuery);

//    QString code = "qt.jQuery('span').remove()";
//    view->page()->mainFrame()->evaluateJavaScript(code);

    if(m_isContent) {
        QtConcurrent::run(HTML2Text,view->page()->mainFrame()->toHtml(),m_currentURL);
//        QString html(view->page()->mainFrame()->toHtml());
//        qDebug()<< view->page()->mainFrame()->toHtml().remove(QRegularExpression("<span((.|\n)*)\/span>"));
//        QProcess process;
//        process.start("node /home/dyz/MLProj/dataCollect/reada.js");
//        process.waitForStarted();
//        qDebug() << view->page()->mainFrame()->toHtml();
//        (<span[^>]*>((.|\n)*?)<\/span>|<\/span>)
//        <span[^>]*>((.|\n)*?)<\/span>

//        process.write(view->page()->mainFrame()->toHtml()./*remove(QRegularExpression("(<span[^>]*>((.|\n)*?)<\/span>|<\/span>)")).*/toStdString().c_str());
//        process.closeWriteChannel();
//        process.waitForFinished(-1); // will wait forever until finished

//        QString stdout = process.readAllStandardOutput();
//        QString stderr = process.readAllStandardError();
//        qDebug() << m_currentURL;
//        qDebug() << stdout.remove(QRegularExpression("(<!--((.|\n)*?)-->|<[^>]*>|»|\n)"));;
//        qDebug() << stderr;
        if (m_hrefs.isEmpty()) m_isContent = false;
        else {
            m_currentURL = m_hrefs.first();
            m_hrefs.pop_front();
            view->load(m_currentURL);
        }
        return;
    }
    view->setFocus();


    QWebElementCollection collection = view->page()->mainFrame()->findAllElements("div#archivedArticles");
    QWebElement element = collection.first();
    QWebElementCollection coll = element.findAll("a");
    foreach(QWebElement ele, coll)
    {
        QString href = ele.attribute("href");
        if (!href.isEmpty())
        {
            qDebug() << href;
            m_isContent = true;
            m_hrefs.push_back(href);
        }
    }
    //m_hrefs.push_front("http://online.wsj.com/news/articles/SB10001424052748704594804575648903868068536");
    if (m_isContent) {
        m_currentURL = m_hrefs.first();
        m_hrefs.pop_front();
        view->load(m_currentURL);
    }

//    rotateImages(rotateAction->isChecked());
}
//! [6]

//! [7]
void MainWindow::highlightAllLinks()
{
    // We append '; undefined' after the jQuery call here to prevent a possible recursion loop and crash caused by
    // the way the elements returned by the each iterator elements reference each other, which causes problems upon
    // converting them to QVariants.
    QString code = "qt.jQuery('a').each( function () { qt.jQuery(this).css('background-color', 'yellow') } ); undefined";
    view->page()->mainFrame()->evaluateJavaScript(code);
}
//! [7]

//! [8]
void MainWindow::rotateImages(bool invert)
{
    QString code;

    // We append '; undefined' after each of the jQuery calls here to prevent a possible recursion loop and crash caused by
    // the way the elements returned by the each iterator elements reference each other, which causes problems upon
    // converting them to QVariants.
    if (invert)
        code = "qt.jQuery('img').each( function () { qt.jQuery(this).css('-webkit-transition', '-webkit-transform 2s'); qt.jQuery(this).css('-webkit-transform', 'rotate(180deg)') } ); undefined";
    else
        code = "qt.jQuery('img').each( function () { qt.jQuery(this).css('-webkit-transition', '-webkit-transform 2s'); qt.jQuery(this).css('-webkit-transform', 'rotate(0deg)') } ); undefined";
    view->page()->mainFrame()->evaluateJavaScript(code);
}
//! [8]

//! [9]
void MainWindow::removeGifImages()
{
//    QString code = "qt.jQuery('[src*=gif]').remove()";
//    view->page()->mainFrame()->evaluateJavaScript(code);
        view->load(QUrl("http://online.wsj.com/public/page/archive-2010-10-1.html"));

}

void MainWindow::removeInlineFrames()
{
    QString code = "qt.jQuery('iframe').remove()";
    view->page()->mainFrame()->evaluateJavaScript(code);
}

void MainWindow::removeObjectElements()
{
    QString code = "qt.jQuery('object').remove()";
    view->page()->mainFrame()->evaluateJavaScript(code);
}

void MainWindow::removeEmbeddedElements()
{
    QString code = "qt.jQuery('embed').remove()";
    view->page()->mainFrame()->evaluateJavaScript(code);
}
//! [9]

