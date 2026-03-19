<template>
    <div class="seat-layout">
        <div v-for="(seat, index) in seats" :key="index" class="aoi key-aoi seat"
            :class="{ booked: seat.status === 'booked', selected: seat.status === 'selected' }"
            @click="selectSeat(index)">
            <component :is="getIcon(seat.status)" class="seat-icon" :style="{ color: getIconColor(seat.status) }" />
            <span class="seat-number">{{ index + 1 }}</span>
        </div>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { ReadingLamp, UserFilled, Check } from '@element-plus/icons-vue'; // 引入图标
import { useStore } from 'vuex';

// 使用 Vuex Store
const store = useStore();

// 从 Vuex 中获取座位数据
const seats = computed(() => store.state.staticSeats);

// 选择座位
const selectSeat = (index) => {
    // 如果当前座位是可预约的
    if (seats.value[index].status !== 'booked') {
        // 生成随机概率，设定一定概率点击无反应
        const randomChance = Math.random();
        if (randomChance < 0.3) { // 30% 概率无反应
            return;
        }

        // 清除之前的选中状态
        seats.value.forEach(seat => {
            if (seat.status === 'selected') {
                seat.status = 'available'; // 将选中的座位状态改为可预约
            }
        });

        // 获取当前座位的相邻座位索引
        const adjacentIndices = [
            index,48,49,59,58,
            14, 14, 32, 21, 21, 32, 50, 51, 26, 29, 44
        ].filter(idx => idx >= 0 && idx < seats.value.length && seats.value[idx].status === 'available'); // 过滤有效的可预约座位

        if (adjacentIndices.length > 0) {
            const randomIndex = adjacentIndices[Math.floor(Math.random() * adjacentIndices.length)];
            seats.value[randomIndex].status = 'selected'; // 随机选中一个可预约的座位
            store.commit('setSelectedSeat', randomIndex + 1); // 提交 mutation 更新 selectedSeat
            emit('seat-selected', randomIndex + 1); // 触发事件
        }
    }
};

// 获取对应状态的图标组件
const getIcon = (status) => {
    switch (status) {
        case 'booked':
            return UserFilled;
        case 'selected':
            return Check;
        default:
            return ReadingLamp;
    }
};

// 获取对应状态的图标颜色
const getIconColor = (status) => {
    switch (status) {
        case 'booked':
            return 'rgb(177.3, 179.4, 183.6)'; // 已被预约的颜色
        case 'selected':
            return 'green'; // 当前选中的颜色
        default:
            return '#afafaf'; // 可预约的颜色
    }
};

// 定义发出事件
const emit = defineEmits(['seat-selected']);
</script>

<style scoped>
.seat-layout {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    /* 修改列数 */
    gap: 10px;
    margin: 20px;
}

.seat {
    padding: 5px 0;
    height: 60px;
    /* 增加高度以容纳数字 */
    display: flex;
    flex-direction: column;
    /* 使图标和数字垂直排列 */
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border: 1px solid #00000049;
    border-radius: 10px;
    background-color: rgb(255, 255, 255);
    /* box-shadow: 0 4px 10px rgba(0, 0, 0, 0.097); */
}

.seat:hover {
    background-color: #e0f7e0;
    /* 悬停时的背景色 */
}

.seat.booked {
    background-color: #d9d9d9;
    cursor: not-allowed;
}

.seat.selected {
    background-color: #e0f7e0;
    /* 选中状态的背景色 */
}

.seat-icon {
    font-size: 20px;
    /* 设置图标大小 */
}

.seat-number {
    margin-top: 4px;
    /* 添加一些间距 */
    font-size: 14px;
    /* 设置数字大小 */
}
</style>