<template>
    <div class="aoip key-aoi alert-message">
        <span class="alert-icon">⚠️</span>
        Please review your reservation details carefully
    </div>
    <div class="info-box">
        <div class="info-container">
            <h2 class="aoim key-aoi">Reservation Details</h2>
            <p class="aoim key-aoi"><strong>Floor:</strong> {{ selectedFloor }}</p>
            <p class="aoim key-aoi"><strong>Date:</strong> {{ formattedDate }}</p>
            <p class="aoim key-aoi"><strong>Time:</strong> {{ startTime }} - {{ endTime }}</p>
            <p class="aoim key-aoi"><strong>Seat:</strong> {{ selectedSeat }}</p>
        </div>
    </div>
</template>

<script setup>
import { computed } from 'vue';
import { useStore } from 'vuex';

const store = useStore();

const selectedFloor = computed(() => store.state.selectedFloor);
const startTime = computed(() => store.state.startTime);
const endTime = computed(() => store.state.endTime);
const selectedSeat = computed(() => store.state.selectedSeat);
const selectedDate = computed(() => store.state.selectedDate);

const formattedDate = computed(() => {
    const date = new Date(selectedDate.value);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
});
</script>

<style scoped>
.alert-message {
    margin-top: 1rem;
    padding: 10px;
    background-color: #ddf1ff;
    color: #0077d8;
    border: 1px solid #0097d8;
    border-radius: 5px;
    display: flex;
    align-items: center;
    margin: 0 400px;
}

.alert-icon {
    margin-right: 8px;
    color: #0090d8;
    font-size: 1.5rem;
}

.info-box {
    display: flex;
    justify-content: center;
    margin: 20px;
}

.info-container {
    background-color: #0000000d;
    border: 1px solid #c4c4c4;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    width: 320px;
    text-align: left;
    transition: transform 0.3s;
}


h2 {
    margin-bottom: 15px;
    color: #0277bd;
    font-family: 'Arial', sans-serif;
    font-weight: bold;
}

p {
    margin: 8px 0;
    font-size: 18px;
    color: #c7c6c65c;
    line-height: 1.5;
    user-select: text !important;
}

/* p:hover{
    color: #383838a9;
} */

strong {
    color: #d7d9dba4;
    user-select: text !important;
}
</style>