import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, staticImageMode = False, maxNumHands=2, modelComplexity=1, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
        self.staticImageMode = staticImageMode
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.staticImageMode,self.maxNumHands, self.modelComplexity, self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                      self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                      self.mp_drawing_styles.DrawingSpec((255, 0, 0)))
        return img


    def findPosition(self,img, handNo=0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                lmList.append([id,cx,cy])
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList= detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()