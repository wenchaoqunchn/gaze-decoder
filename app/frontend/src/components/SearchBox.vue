<template>
    <div class="search-container">
        <div class="button-group">
            <button v-for="(item, index) in buttons" :key="index" :class="{ 'active': selectedButton === item }"
                class="aoim filter-button" @click="selectButton(item)">
                {{ item }}
            </button>
        </div>
        <div class="sbox">
            <input class="aoim sinput"></input>
            <button class="aoim sbtn">Search</button>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue';

const buttons = ['Library Catalog', 'E-Journals', 'Articles'];
const selectedButton = ref('Library Catalog'); // default selected button
const selectButton = (item) => {
    selectedButton.value = item; // 更新选中的按钮
};
</script>

<style scoped>
.search-container {
    background-color: rgba(0, 0, 0, 0.284);
    /* 半透明黑色背景 */
    padding: 60px 200px;
    /* border-radius: 8px; */
    /* 最大宽度 */
    margin: 30px 230px;
    /* 居中 */

}

.sbox {
    height: 60px;
    display: flex;
}

.sinput {
    flex: 9;
    margin: 5px auto;
    border-radius: 50px;
    border: none;
    font-size: 16px;
    padding: 5px 20px;
}

.sbtn {
    flex: 1;
    height: 50px;
    margin: auto;
    margin-left: 20px;
    border-radius: 50px;
    border: none;
    background-color: orange;
    font-size: 20px;
    font-weight: bold;
    color: white;
}

.button-group {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 10px;
    /* 按钮左对齐 */
    /* margin: 20px; */
    /* 按钮组的外边距 */
}

.filter-button {
    margin-right: 10px;
    /* 按钮之间的间距 */
    border-radius: 500px;
    /* 圆角按钮 */
    font-size: 20px;
    /* 字体大小 */
    color: white;
    /* 文字颜色 */
    background-color: transparent;
    /* 默认无背景 */
    border: none;
    /* 默认无边框 */
    padding: 10px 40px;
    /* 内边距 */
    cursor: pointer;
    /* 鼠标样式 */
    /* transition: background-color 0.3s, border 0.3s; */
    /* 背景颜色和边框过渡效果 */
}

.filter-button.active {
    background-color: orange;
    /* 选中按钮颜色 */
    border: none;
    font-weight: bold;
    /* 选中按钮无边框 */
}
</style>