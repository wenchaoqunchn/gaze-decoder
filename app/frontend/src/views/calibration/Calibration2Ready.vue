<template>
    <div class="calibration-container">
        <div class="aoi key-aoi target-point" :style="{ top: targetY + 'px', left: targetX + 'px' }">+</div>
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();
let timeout = null;

const targetX = ref(0);
const targetY = ref(0);
const baseLength = 600; // 底边长度
const height = 200; // 高度（可以根据需要调整为较小的值）

const setRandomTarget = (vertexIndex) => {
    const baseX = window.innerWidth / 2; // 屏幕中央X坐标
    const baseY = window.innerHeight / 2; // 屏幕中央Y坐标

    switch (vertexIndex) {
        case 0: // 第一个顶点（顶部）
            targetX.value = baseX; // 顶点坐标X
            targetY.value = baseY - height; // 顶点坐标Y
            break;
        case 1: // 第二个顶点（左下）
            targetX.value = baseX - (baseLength / 2); // 顶点坐标X
            targetY.value = baseY + (height / 2); // 顶点坐标Y
            break;
        case 2: // 第三个顶点（右下）
            targetX.value = baseX + (baseLength / 2); // 顶点坐标X
            targetY.value = baseY + (height / 2); // 顶点坐标Y
            break;
        default:
            break;
    }
};

const startTimer = () => {
    timeout = setTimeout(() => {
        goToNext();
    }, 500); // 3秒后跳转
};

const goToNext = () => {
    clearTimeout(timeout);
    router.push({name:'Calibration2'}); // 根据你的路由设置进行修改
};

onMounted(() => {
    setRandomTarget(1);
    startTimer();
});

onBeforeUnmount(() => {
    clearTimeout(timeout);
});
</script>

<style scoped>
.calibration-container {
    width: 100vw;
    height: 100vh;
    background-color: #363636;
    position: relative;
    overflow: hidden;
    cursor: none;
}

.target-point {
    font-size: 30px;
    /* 调整字体大小 */
    color: rgb(100, 236, 8);
    /* 字体颜色 */
    position: absolute;
    text-align: center;
    transform: translate(-50%, -50%);
    /* 使目标点居中 */

}
</style>