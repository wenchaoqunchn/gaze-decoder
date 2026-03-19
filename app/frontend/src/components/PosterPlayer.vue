<template>
    <div class="aoi carousel-container">
        <el-carousel :interval="5000" direction="horizontal" arrow="always" height="300px" indicator-position="none">
            <el-carousel-item v-for="(item, index) in carouselItems" :key="index">
                <div class="carousel-item">
                    <img :src="item.image" :alt="item.alt" class="carousel-image" />
                    <div class="carousel-caption">{{ item.caption }}</div>
                </div>
            </el-carousel-item>
        </el-carousel>
    </div>
</template>

<script>
import fuwutai from '../assets/player/fuwutai.jpg';
import gujishuku from '../assets/player/gujishuku.jpg';
import tushuguanwaijing2 from '../assets/player/tushuguanwaijing2.jpg';
import yuelanqu from '../assets/player/yuelanqu.jpg';
import zhongting from '../assets/player/zhongting.jpg';
import zixiushi from '../assets/player/zixiushi.jpg';

export default {
    name: 'PosterPlayer',
    data() {
        return {
            carouselItems: [
                { image: fuwutai, alt: 'Service Desk', caption: 'Service Desk' },
                { image: zhongting, alt: 'Central Atrium', caption: 'Central Atrium' },
                { image: zixiushi, alt: 'Study Room', caption: 'Study Room' },
                { image: yuelanqu, alt: 'Reading Area', caption: 'Reading Area' },
                { image: tushuguanwaijing2, alt: 'Library Exterior', caption: 'Library Exterior' },
                { image: gujishuku, alt: 'Special Collections Room', caption: 'Special Collections Room' }
            ],
        };
    },
};
</script>

<style scoped>
.carousel-container {
    max-width: 700px;
    /* 最大宽度 */
    margin: auto;
    /* 居中 */
    position: relative;
    /* 为了定位箭头 */
}

.carousel-item {
    position: relative;
    /* 使 caption 和箭头能够绝对定位 */
}

.carousel-image {
    width: 100%;
    /* 图片宽度自适应 */
    height: 300px;
    /* 固定高度 */
    object-fit: cover;
    /* 保持宽高比，裁剪多余部分 */
}

.carousel-caption {
    position: absolute;
    bottom: 12px;
    /* 距离底部 */
    left: 5px;
    /* 距离左边 */
    color: white;
    /* 文字颜色 */
    background-color: rgba(0, 0, 0, 0.5);
    /* 半透明背景 */
    padding: 10px;
    /* 内边距 */
    border-radius: 5px;
    /* 圆角 */
}

.el-carousel__arrow {
    background: rgba(255, 255, 255, 0.142);
    /* 箭头背景 */
    border-radius: 50%;
    /* 圆形箭头 */
    padding: 10px;
    /* 内边距 */
    z-index: 10;
    /* 确保箭头在图片上方 */
}
</style>