function getAllAOIInfo() {
    // 获取所有 AOI 和关键 AOI 元素
    const elements = document.querySelectorAll('.aoim, .aoip, .aoi, .key-aoi');
    const info = [];

    elements.forEach(element => {
        const rect = element.getBoundingClientRect();
        const computedStyle = window.getComputedStyle(element);

        // 初始化坐标
        let innerTopLeft = { x: rect.left, y: rect.top };
        let innerBottomRight = { x: rect.right, y: rect.bottom };

        // 根据类名调整坐标
        if (element.classList.contains('aoim')) {
            const marginLeft = parseFloat(computedStyle.marginLeft);
            const marginTop = parseFloat(computedStyle.marginTop);
            innerTopLeft.x -= marginLeft;
            innerTopLeft.y -= marginTop;
            innerBottomRight.x += parseFloat(computedStyle.marginRight);
            innerBottomRight.y += parseFloat(computedStyle.marginBottom);
        } else if (element.classList.contains('aoip')) {
            const paddingLeft = parseFloat(computedStyle.paddingLeft);
            const paddingTop = parseFloat(computedStyle.paddingTop);
            innerTopLeft.x += paddingLeft;
            innerTopLeft.y += paddingTop;
            innerBottomRight.x -= parseFloat(computedStyle.paddingRight);
            innerBottomRight.y -= parseFloat(computedStyle.paddingBottom);
        }

        // 判断是否为关键 AOI
        const isKeyAOI = element.classList.contains('key-aoi');

        // 组件信息，可以根据需要修改
        const componentInfo = element.className;

        innerTopLeft.x = innerTopLeft.x * 1.5
        innerTopLeft.y = innerTopLeft.y * 1.5 + 124
        innerBottomRight.x = innerBottomRight.x * 1.5
        innerBottomRight.y = innerBottomRight.y * 1.5 + 124

        info.push({
            topLeft: innerTopLeft,
            bottomRight: innerBottomRight,
            isKeyAOI: isKeyAOI,
            componentInfo: componentInfo
        });
    });

    return info;
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 导出所有 AOI 数据为 JSON 文件
function exportAllAOIInfo(filename) {
    const info = getAllAOIInfo();
    downloadJSON(info, filename + '.json'); // 取消注释以启用下载
    console.log(info);
}

export { exportAllAOIInfo };