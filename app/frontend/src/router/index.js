import { createRouter, createWebHistory } from 'vue-router';

import MainLayout from '../views/MainLayout.vue';
import HomePage from '../views/HomePage.vue';
import SessionReady from '../views/SessionReady.vue';
import SessionDone from '../views/SessionDone.vue';

import Overview from '../views/overview/Overview.vue';
import LeaderSpeech from '../views/overview/LeaderSpeech.vue';
import LibIntro from '../views/overview/LibIntro.vue';
import LibLayout from '../views/overview/LibLayout.vue';
import LibRule from '../views/overview/LibRule.vue';
import ServiceTime from '../views/overview/ServiceTime.vue';
import ServiceOverview from '../views/overview/ServiceOverview.vue';

import ContactUs from '../views/ContactUs.vue';

import Resources from '../views/resources/Resources.vue';
import CoreJournal from '../views/resources/CoreJournal.vue';
import EBook from '../views/resources/EBook.vue';
import LibThesis from '../views/resources/LibThesis.vue';
import CommonApp from '../views/resources/CommonApp.vue';
import Copyright from '../views/resources/Copyright.vue';

import ServicePage from '../views/services/ServicePage.vue';
import BookBorrow from '../views/services/BookBorrow.vue';
import CardProcess from '../views/services/CardProcess.vue';
import AncientRead from '../views/services/AncientRead.vue';
import DiscRequest from '../views/services/DiscRequest.vue';
import DocumentTransfer from '../views/services/DocumentTransfer.vue';
import TechSearch from '../views/services/TechSearch.vue';
import InfoTeaching from '../views/services/InfoTeaching.vue';
import SeatReserve from '../views/services/SeatReserve.vue';
import VolunteerTeam from '../views/services/VolunteerTeam.vue';

import Reserve from '../views/reserve/Reserve.vue';
import FloorSelect from '../views/reserve/FloorSelect.vue';
import TimeSelect from '../views/reserve/TimeSelect.vue';
import SeatSelect from '../views/reserve/SeatSelect.vue';
import InfoConfirm from '../views/reserve/InfoConfirm.vue';

import Calibration1Ready from '../views/calibration/Calibration1Ready.vue';
import Calibration1 from '../views/calibration/Calibration1.vue';
import Calibration2Ready from '../views/calibration/Calibration2Ready.vue';
import Calibration2 from '../views/calibration/Calibration2.vue';
import Calibration3Ready from '../views/calibration/Calibration3Ready.vue';
import Calibration3 from '../views/calibration/Calibration3.vue';


import axios from 'axios';
import { useStore } from 'vuex';
import { exportAllAOIInfo } from '../getAOIInfo';
import { ElMessage } from 'element-plus';

const routes = [
    {
        path: '/',
        name: 'SessionReady',
        component: SessionReady
    },
    {
        path: '/calibration',
        redirect: '/calibration/1',
        children: [
            {
                path: 'c1',
                name: 'Calibration1',
                component: Calibration1
            },
            {
                path: 'c2',
                name: 'Calibration2',
                component: Calibration2
            },
            {
                path: 'c3',
                name: 'Calibration3',
                component: Calibration3
            },
            {
                path: 'r1',
                name: 'Calibration1Ready',
                component: Calibration1Ready
            },
            {
                path: 'r2',
                name: 'Calibration2Ready',
                component: Calibration2Ready
            },
            {
                path: 'r3',
                name: 'Calibration3Ready',
                component: Calibration3Ready
            }
        ]
    },
    {
        path: '/home',
        component: MainLayout,
        children: [
            {
                path: '', // 默认子路由
                name: 'HomePage',
                component: HomePage,
            },
            {
                path: '/overview',
                name: 'Overview',
                component: Overview,
                redirect: '/overview/lib-intro',
                children: [
                    {
                        path: 'leader-speech',
                        name: 'LeaderSpeech',
                        component: LeaderSpeech,
                    },
                    {
                        path: 'lib-intro',
                        name: 'LibIntro',
                        component: LibIntro,
                    },
                    {
                        path: 'lib-layout',
                        name: 'LibLayout',
                        component: LibLayout,
                    },
                    {
                        path: 'lib-rule',
                        name: 'LibRule',
                        component: LibRule,
                    },
                    {
                        path: 'service-time',
                        name: 'ServiceTime',
                        component: ServiceTime,
                    },
                    {
                        path: 'service-overview',
                        name: 'ServiceOverview',
                        component: ServiceOverview,
                    }
                ],
            },
            {
                path: '/resources',
                name: 'Resources',
                redirect: '/resources/core-journal',
                component: Resources,
                children: [
                    {
                        path: 'core-journal',
                        name: 'CoreJournal',
                        component: CoreJournal,
                    },
                    {
                        path: 'e-book',
                        name: 'EBook',
                        component: EBook,
                    },
                    {
                        path: 'lib-thesis',
                        name: 'LibThesis',
                        component: LibThesis,
                    },
                    {
                        path: 'common-app',
                        name: 'CommonApp',
                        component: CommonApp,
                    },
                    {
                        path: 'copyright',
                        name: 'Copyright',
                        component: Copyright,
                    }
                ],
            },
            {
                path: '/services',
                name: 'Services',
                component: ServicePage,
                redirect: '/services/book-borrow',
                children: [
                    {
                        path: 'book-borrow',
                        name: 'BookBorrow',
                        component: BookBorrow,
                    },
                    {
                        path: 'card-process',
                        name: 'CardProcess',
                        component: CardProcess,
                    },
                    {
                        path: 'ancient-read',
                        name: 'AncientRead',
                        component: AncientRead,
                    },
                    {
                        path: 'disc-request',
                        name: 'DiscRequest',
                        component: DiscRequest,
                    },
                    {
                        path: 'document-transfer',
                        name: 'DocumentTransfer',
                        component: DocumentTransfer,
                    },
                    {
                        path: 'tech-search',
                        name: 'TechSearch',
                        component: TechSearch,
                    },
                    {
                        path: 'info-teaching',
                        name: 'InfoTeaching',
                        component: InfoTeaching,
                    },
                    {
                        path: 'seat-reserve',
                        name: 'SeatReserve',
                        component: SeatReserve,
                    },
                    {
                        path: 'volunteer-team',
                        name: 'VolunteerTeam',
                        component: VolunteerTeam,
                    },
                ],
            },
            {
                path: '/reserve',
                name: 'Reserve',
                component: Reserve,
                redirect: '/reserve/floor-select',
                children: [
                    {
                        path: 'floor-select',
                        name: 'FloorSelect',
                        component: FloorSelect,
                    },
                    {
                        path: 'time-select',
                        name: 'TimeSelect',
                        component: TimeSelect,
                    },
                    {
                        path: 'seat-select',
                        name: 'SeatSelect',
                        component: SeatSelect,
                    },
                    {
                        path: 'info-confirm',
                        name: 'InfoConfirm',
                        component: InfoConfirm,
                    },
                ]
            },
            {
                path: '/contact',
                name: 'ContactUs',
                component: ContactUs
            }
        ]
    },
    {
        path: '/done',
        name: 'SessionDone',
        component: SessionDone
    }
];

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes
});

router.beforeEach(async (to, from, next) => {
    const store = useStore();

    // 判断是否需要进行截图
    if (to.name !== 'SessionReady' && from.name !== 'SessionReady') {
        if (store.state.exportAOI) {
            exportAllAOIInfo(from.name);
        }
        if (store.state.skipRequest) {
            ElMessage.success('API request skipped');
            next();
            return;
        }
        try {
            // 发送截图请求
            await axios.post('/api/switch', { currentPage: from.name });
            console.log('Screenshot request sent for:', from.name);
        } catch (error) {
            console.error('Screenshot request failed:', error);
        }
    }
    next(); // 确保调用 next() 继续路由导航
});

export default router;