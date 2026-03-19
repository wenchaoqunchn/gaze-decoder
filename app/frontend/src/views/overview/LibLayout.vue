<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > About the Library > Floor Layout
        </div>
        <div class="aoi title">
            <h2>Floor Layout</h2>
        </div>
        <div class="aoi content">
            <el-carousel height="65vh" type="card" arrow="always" indicator-position="none" :autoplay="false">

                <el-carousel-item>
                    <img src="../../assets/layout/f2.jpg" alt="Library layout level 2" class="carousel-image" />
                </el-carousel-item>
                <el-carousel-item>
                    <img src="../../assets/layout/f3.jpg" alt="Library layout level 3" class="carousel-image" />
                </el-carousel-item>
                <el-carousel-item>
                    <img src="../../assets/layout/f4.jpg" alt="Library layout level 4" class="carousel-image" />
                </el-carousel-item>
            </el-carousel>
        </div>
    </div>
</template>

<script setup>
// Import Element Plus components
import { ElCarousel, ElCarouselItem } from 'element-plus';
</script>

<style scoped>
.title {
    text-align: center;
}

.main-content {
    padding: 20px 5%;
}

.content {
    font-size: 18px;
    line-height: 1.6;
}

.carousel-image {
    width: auto;
    height: 100%;
    object-fit: cover;
    /* Keep image proportions */
}
</style>