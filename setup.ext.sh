# REPO="dessant/buster"
# LATEST_VERSION=$(curl -s https://api.github.com/repos/${REPO}/releases | grep -m 1 "tag_name" | cut -d'"' -f4)
# LATEST_VERSION_V=${LATEST_VERSION#v}
# FILE_NAME="buster_captcha_solver_for_humans-${LATEST_VERSION_V}-chrome.zip"
# FILE_NAME_O="buster.zip"
# echo "最新版本是: ${LATEST_VERSION}"
# curl -L "https://github.com/${REPO}/releases/download/${LATEST_VERSION}/${FILE_NAME}" -o ${FILE_NAME_O}
# unzip -o ${FILE_NAME_O} -d buster

# https://www.crxsoso.com/webstore/detail/kahepbmokpbbkfognbmimpmnlfibcolf
unzip -o "hCaptcha Solver by NoneCap 0.3.100.zip" -d hCaptchaSolver