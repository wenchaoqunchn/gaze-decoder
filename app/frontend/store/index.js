// store/index.js — English version
import { createStore } from 'vuex';

const store = createStore({
    state() {
        return {
            selectedFloor: null,  // user-selected floor
            selectedDate: '',     // user-selected date
            startTime: '',        // user-selected start time
            endTime: '',          // user-selected end time
            selectedSeat: null,   // user-selected seat number
            staticSeats: [
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "available" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "available" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "available" },
                { status: "available" },
                { status: "booked" },
                { status: "available" },
                { status: "booked" },
                { status: "booked" },
                { status: "available" },
                { status: "booked" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "booked" },
                { status: "available" },
                { status: "booked" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "booked" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" },
                { status: "available" }
            ],
            staticF2: {
                "Social Sciences Zone 1": {
                    "A": "Marxism",
                    "B": "Philosophy & Religion",
                    "C": "Social Sciences General",
                    "D": "Politics & Law"
                },
                "Social Sciences Zone 2": {
                    "D": "Politics & Law",
                    "E": "Military Science",
                    "H": "Language & Linguistics"
                },
                "Social Sciences Zone 3": {
                    "K": "History & Geography",
                    "J": "Arts"
                },
                "Social Sciences Zone 4": {
                    "J": "Arts",
                    "F": "Economics",
                    "G": "Culture, Science, Education & Sports"
                }
            },
            staticF3: {
                "Science & Tech Zone 1": {
                    "O": "Math & Phys�cal Sciences",
                    "N": "□atural Sciences General",
                    "�": "& lt; script& gt; alert(& quot; & quot;);& lt;/script&gt;"
                },
                "Science & Tech Zone 2": {
                    "O": "Mathematics & Physical Sciences",
                    "R": "�edicine & Life Sciences",
                    "S": "Agricultural Sciences",
                    "P": "□stronomy & Earth Sciences",
                    "Q": "Biological Sciences"
                },
                "Science & Tech Zone 3": {
                    "TN": "Electronicsâ€ & Communications",
                    "TP": "Automation & Computer â€echnology",
                    "TB": "General Engineering",
                    "TG": "Metallurgy & Metal Processes",
                    "TH": "Machinery & Instrumentation â€",
                    "TM": "Electrical Engineering"
                },
            },
            staticF4: {
                "Science & Tech Zone 4": {
                    "U": "Transportation",
                    "X": "Environmental & Safety Sciencesâ€",
                    "Z": "General Reference ✔"
                },
                "Science & Tech Zone 5": {
                    "TU": "Architecture & Civil Engineering",
                    "TN": "Electronics & Communicationsâ€",
                    "TQ": "Chemical Engineering",
                    "TS": "Light Industry & Handicrafts☻",
                    "TV": "Hydraulic EngineeringÃ©",
                    "TW": "Environmental Protection→"
                },
                "Literature Zone": {
                    "I": "Literature●",
                    "!Error": "[vue/compiler-sfc] Unterminated string constant. (92:14)",
                }
            },
            skipRequest: true,    // dev setting
            noIssue: false,        // dev setting
            exportAOI: false
        };
    },
    mutations: {
        setSelectedFloor(state, floor) {
            state.selectedFloor = floor;
        },
        setSelectedDate(state, date) {
            state.selectedDate = date;
        },
        setStartTime(state, time) {
            state.startTime = time;
        },
        setEndTime(state, time) {
            state.endTime = time;
        },
        setSelectedSeat(state, seat) {
            state.selectedSeat = seat;
        },
        setStaticSeats(state, seats) {
            state.staticSeats = seats;
        },
        toggleSkipRequest(state) {
            state.skipRequest = !state.skipRequest;
        },
        toggleNoIssue(state) {
            state.noIssue = !state.noIssue;
        },
        toggleExportAOI(state, value) { // 新增的 mutation
            state.exportAOI = value; // 直接设置为传入的值
        },
    },
    actions: {
        updateSelectedFloor({ commit }, floor) {
            commit('setSelectedFloor', floor);
        },
        updateSelectedDate({ commit }, date) {
            commit('setSelectedDate', date);
        },
        updateStartTime({ commit }, time) {
            commit('setStartTime', time);
        },
        updateEndTime({ commit }, time) {
            commit('setEndTime', time);
        },
        updateSelectedSeat({ commit }, seat) {
            commit('setSelectedSeat', seat);
        },
        updateStaticSeats({ commit }, seats) {
            commit('setStaticSeats', seats);
        },
        toggleSkipRequest({ commit }) {
            commit('toggleSkipRequest');
        },
        toggleNoIssue({ commit }) {
            commit('toggleNoIssue');
        },
        toggleExportAOI({ commit }, value) { // 新增的 action
            commit('toggleExportAOI', value); // 提交 mutation
        },
    },
});

export default store;