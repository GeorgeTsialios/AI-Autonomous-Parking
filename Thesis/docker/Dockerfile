FROM dalibo/pandocker:latest

# Proxy to APT cacher: e.g. http://apt-cacher-ng.docker:3142
ARG APT_CACHER

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV DEBCONF_NOWARNINGS yes

# Debian
#
RUN set -x && \
    # Setup a cacher to speed up build
    if [ -n "${APT_CACHER}" ] ; then \
        echo "Acquire::http::Proxy \"${APT_CACHER}\";" | tee /etc/apt/apt.conf.d/01proxy ; \
    fi; \
    apt-get -qq update && \
    apt-get -qy install --no-install-recommends \
        #
        texlive-lang-other \
        # hindi fonts
        #fonts-deva \
        # persian fonts
        #texlive-lang-arabic \
        #fonts-farsiweb \
        # dia
        #dia \
        # Noto font families with large Unicode coverage
        fonts-noto \
        fonts-noto-cjk \
        fonts-noto-cjk-extra \
        fonts-noto-color-emoji \
        fonts-noto-core \
        fonts-noto-extra \
        fonts-noto-mono \
        fonts-linuxlibertine \
        texlive-bibtex-extra \
        biber \
        librsvg2-bin \
        gcc \
        python3-dev \
        latex-cjk-all \
    # clean up
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/01proxy

##
## M I S C
##

#
# emojis support for latex
# https://github.com/mreq/xelatex-emoji
#
ARG TEXMF=/usr/share/texmf/tex/latex/
ARG EMOJI_DIR=/tmp/twemoji2
RUN git clone --single-branch --depth=1 --branch gh-pages https://github.com/twitter/twemoji.git $EMOJI_DIR && \
    # fetch xelatex-emoji
    mkdir -p ${TEXMF} && \
    cd ${TEXMF} && \
    rm -rf xelatex-emoji2 && \
    git clone --single-branch --branch images https://github.com/daamien/xelatex-emoji.git xelatex-emoji2 && \
    # convert twemoji SVG files into PDF files
    cp -r $EMOJI_DIR/2/svg xelatex-emoji2/images && \
    cd xelatex-emoji2/images && \
    ../bin/convert_svgs_to_pdfs ./*.svg && \
    # clean up
    rm -f *.svg && \
    rm -fr ${EMOJI_DIR} && \
    rm -rf xelatex-emoji2 && \
    # update texlive
    cd ${TEXMF} && \
    texhash

RUN pip3 install pandoc-fignos
RUN pip3 install pandoc-secnos
RUN pip3 install pandoc-tablenos
RUN pip3 install pandoc-xnos
RUN apt install latex-cjk-all
RUN apt remove -y gcc python3-dev
RUN apt autoremove -y
RUN apt autoclean -y
# RUN rm -rf /usr/share/texmf/tex/latex/twemoji*
# RUN rm -rf /usr/share/texmf/fonts/truetype/OpenMoji-Color.ttf
RUN kpsewhich -var-value TEXMFLOCAL 
RUN mkdir -p /usr/local/share/texmf/tex/latex
RUN wget -O  /usr/local/share/texmf/tex/latex/twemojis.zip https://mirrors.ctan.org/macros/latex/contrib/twemojis.zip
RUN unzip    /usr/local/share/texmf/tex/latex/twemojis.zip -d /usr/local/share/texmf/tex/latex/
RUN rm       /usr/local/share/texmf/tex/latex/twemojis.zip
# RUN cd /usr/local/share/texmf/tex/latex/twemojis/ && pdflatex twemojis.ins
RUN wget -O  /usr/local/share/texmf/tex/latex/twemojis/twemojis.sty http://hci.ece.upatras.gr/twemojis.sty

RUN wget -O /tmp/fonts-jetbrains-mono_2.242+ds-2_all.zip 'https://download.jetbrains.com/fonts/JetBrainsMono-2.304.zip'
RUN mkdir -p /usr/share/fonts/truetype/jetbrains-mono
RUN unzip /tmp/fonts-jetbrains-mono_2.242+ds-2_all.zip -d /usr/share/fonts/truetype/jetbrains-mono
RUN rm -f /tmp/fonts-jetbrains-mono*

RUN wget -O /tmp/fonts-ubuntu_0.83-6_all.deb 'http://ftp.br.debian.org/debian/pool/non-free/f/fonts-ubuntu/fonts-ubuntu_0.83-6_all.deb' \
    &&  dpkg -i /tmp/fonts-ubuntu_0.83-6_all.deb
RUN rm -f /tmp/fonts-ubuntu_0.83-6_all.deb

RUN wget -O /tmp/fonts-ubuntu-console_0.83-6_all.deb 'http://ftp.br.debian.org/debian/pool/non-free/f/fonts-ubuntu/fonts-ubuntu-console_0.83-6_all.deb' \
    &&  dpkg -i /tmp/fonts-ubuntu-console_0.83-6_all.deb
RUN rm -f /tmp/fonts-ubuntu-console_0.83-6_all.deb

RUN wget  -O /tmp/fonts-trajan.zip 'https://static.boldfonts.com/download/trajan-pro-font/'
RUN mkdir -p /usr/share/fonts/truetype/trajan
RUN unzip /tmp/fonts-trajan.zip -d /usr/share/fonts/truetype/trajan

RUN mktexlsr

# RUN wget -O /usr/share/texmf/fonts/truetype/OpenMoji-Color.ttf 'https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/font/OpenMoji-Color.ttf'
# RUN mktexlsr
VOLUME /pandoc
WORKDIR /pandoc

ENTRYPOINT ["pandoc"]