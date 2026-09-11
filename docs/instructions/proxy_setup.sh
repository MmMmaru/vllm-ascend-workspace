# for server start by 90
export http_proxy="http://p_atlas:proxy%40123@90.255.24.103:6688"
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# for server start by 80.253
export http_proxy="http://80.253.152.177:7897"
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


# for server start by 141.3
export http_proxy="http://141.3.153.110:7897"
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple