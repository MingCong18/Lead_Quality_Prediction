# Java
sudo yum -y install java-1.8.0-openjdk-devel

# Build Esentials (minimal)
sudo yum -y install gcc gcc-c++ kernel-devel make automake autoconf swig git unzip libtool binutils patch wget bzip2

# Extra Packages for Enterprise Linux (EPEL) (for pip, zeromq3)
sudo yum -y install epel-release

# Python
sudo yum -y install numpy python-devel python-pip
sudo pip install --upgrade pip

# Other TF deps
sudo yum -y install freetype-devel libpng12-devel zip zlib-devel giflib-devel zeromq3-devel
sudo pip install grpcio_tools mock

# HTTP2 Curl
sudo yum -y install libev libev-devel zlib zlib-devel openssl openssl-devel
pushd /var/tmp
git clone https://github.com/tatsuhiro-t/nghttp2.git
cd nghttp2
autoreconf -i
automake
autoconf
./configure
make
sudo make install
sudo echo '/usr/local/lib' > /etc/ld.so.conf.d/custom-libs.conf # if permission denied, write /usr/local/lib to custom-libs.con in /etc/ld.so.conf.d directory

ldconfig
popd

pushd /var/tmp
wget http://curl.haxx.se/download/curl-7.46.0.tar.bz2
tar -xvjf curl-7.46.0.tar.bz2
cd curl-7.46.0
./configure --with-nghttp2=/usr/local --with-ssl
make
sudo make install
sudo ldconfig
popd

# Bazel
pushd /var/tmp
wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-installer-linux-x86_64.sh
chmod +x bazel-*
sudo ./bazel-*
export PATH=/usr/local/bin:$PATH
popd