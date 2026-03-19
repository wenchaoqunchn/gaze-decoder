<!-- AOISelector.vue -->
<template>
    <div @mousemove="updatePreview" class="aoi-selector">
        <slot></slot> <!-- 用于插入页面内容 -->
        <div v-if="previewVisible" class="overlay">
            <div class="highlight" :style="highlightStyle"></div>
        </div>
        <div v-if="hoveredInfo" class="aoi-info"
            :style="{ left: `${hoveredInfo.rect.left}px`, top: `${hoveredInfo.rect.bottom}px` }">
            <p>AOI Info:</p>
            <p>Top Left: (x:{{ Math.round(hoveredInfo.rect.left) }}, y:{{ Math.round(hoveredInfo.rect.top) }})</p>
            <p>Bottom Right: (x:{{ Math.round(hoveredInfo.rect.right) }}, y:{{ Math.round(hoveredInfo.rect.bottom) }})</p>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue';

const selectedAOIs = ref([]);
// const previewVisible = ref(false);
const hoveredTarget = ref(null);
const hoveredInfo = ref(null);
let lastHoveredElement = null; // 用于存储上一个选中的元素

// 选择 AOI 的逻辑
const selectAOI = (event) => {
    previewVisible.value = false; // 隐藏预览

    switch (event.key) {
        case 'a':
            overlayColor.value = 'rgba(189, 241, 177, 0.5)'; // 浅黄色
            break;
        case 's':
            overlayColor.value = 'rgba(255, 177, 177, 0.5)'; // 浅红色
            break;
        case 'd':
            overlayColor.value = 'none'; // 清除覆盖颜色
            break;
    }
};

// 清除选择
const clearSelection = () => {
    selectedAOI.value = null;
    previewVisible.value = false; // 隐藏预览
    if (lastSelectedElement) {
        lastSelectedElement.style.border = ''; // 移除边框
        lastSelectedElement = null; // 清空最后选中的元素
    }
};

// 更新预览
const updatePreview = (event) => {
    const target = event.target; // 获取鼠标悬停的元素
    hoveredTarget.value = target;
    if (target && target !== lastHoveredElement) {
        const rect = target.getBoundingClientRect();
        hoveredInfo.value = {
            rect: {
                left: rect.left + window.scrollX,
                top: rect.top + window.scrollY,
                right: rect.right + window.scrollX,
                bottom: rect.bottom + window.scrollY
            }
        };
    }
};

// 计算高亮层的样式
const highlightStyle = computed(() => {
    if (selectedAOI.value) {
        const { left, top, right, bottom } = selectedAOI.value.rect;
        return {
            position: 'absolute',
            left: `${Math.round(left)}px`,
            top: `${Math.round(top)}px`,
            width: `${Math.round(right - left)}px`,
            height: `${Math.round(bottom - top)}px`,
            backgroundColor: 'rgba(189, 241, 177, 0.5)', // 半透明颜色
            pointerEvents: 'none' // 使覆盖层不阻止点击事件
        };
    }
    return {};
});

// 添加事件监听
onMounted(() => {
    document.addEventListener('keydown', selectAOI);
});

// 移除事件监听以避免内存泄漏
onBeforeUnmount(() => {
    document.removeEventListener('keydown', selectAOI);
});
</script>

<style>
.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    /* 允许点击穿透 */
}

.highlight {
    position: absolute;
    background-color: rgba(189, 241, 177, 0.5);
    /* 半透明的高亮颜色 */
    pointer-events: none;
    /* 允许点击穿透 */
}

.aoi-info {
    position: absolute;
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #ccc;
    padding: 10px;
    z-index: 1000;
}
</style>