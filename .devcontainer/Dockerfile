FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV PATH=${PATH}:/root/.local/bin
ENV HISTFILE=/root/.history/.zsh_history

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt -y install --no-install-recommends apt-utils 2>&1

RUN apt -y dist-upgrade
RUN apt install --no-install-recommends -y \
        build-essential \
        git \
        git-lfs \
        gnupg-agent \
        groff \
        curl \
        dialog \
        wget \
        unzip \
        openssh-client \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \ 
        libyaml-dev \
        apt-transport-https \
        software-properties-common \
        xz-utils \
        zsh \
        pkg-config \
        libhdf5-dev \
        graphviz \
        nvtop \
        libgl1 \
        libaio-dev \
    && apt -y autoremove && apt-get clean -y

# Prepare git lfs in case we need it
RUN git lfs install --skip-smudge

# set zsh as the defaul shell
RUN chsh -s $(which zsh)

# Install Python tools
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python3.11 -m pip install --no-cache-dir -U pip wheel setuptools uv

ARG WORKSPACE_FOLDER
RUN mkdir /workspace
WORKDIR ${WORKSPACE_FOLDER}

# run the oh-my-zsh installation script  
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="kolo"/g' ~/.zshrc

# install powerlevel10k
# RUN git clone https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k

# RUN curl -fsSLO https://raw.githubusercontent.com/romkatv/dotfiles-public/master/.purepower

# Installing cutlass
RUN git clone https://github.com/NVIDIA/cutlass /root/cutlass
ENV CUTLASS_PATH=/root/cutlass

ENV DEBIAN_FRONTEND=dialog
