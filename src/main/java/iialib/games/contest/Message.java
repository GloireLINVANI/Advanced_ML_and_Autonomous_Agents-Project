package iialib.games.contest;

import java.util.StringTokenizer;

public class Message {

    enum Type {
        BEST_MOVE("BEST_MOVE")
        , ROLE("ROLE")
        , TEAM_NAME("TEAM_NAME")
        , I_PLAY("I_PLAY")
        , OTHER_PLAY("OTHER_PLAY")
        , VICTORY("VICTORY")
        , DEFEAT("DEFEAT")
        , TIE("TIE")
        // Pour affiner les informations de fins de partie renvoyées dans la console des joueurs
        // , TIMEOUT("TIMEOUT")
        // , ILLEGAL_MOVE("ILLEGAL MOVE")
        ;
        final String name;
        Type(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return name;
        }
    }
    final Type  type;
    final String msg;
    public static final String END = "\0";
    public static final String SEP = " ";

    Message(String message){
        StringTokenizer tokenizer = new StringTokenizer( message, SEP+END+"\n");
        assert tokenizer.hasMoreElements();
        type = Type.valueOf(tokenizer.nextToken());
        if (type == Type.TEAM_NAME) {
            msg = buildTeamName(tokenizer);
        } else {
            if (tokenizer.hasMoreElements())
                msg = tokenizer.nextToken();
            else
                msg = "";
        }
    }

    Message(Type type, String msg){
        this.type = type;
        this.msg = msg;
    }

    Message(Type type){
        this.type = type;
        this.msg = "";
    }

    private String buildTeamName(StringTokenizer tokenizer){
        StringBuilder sb = new StringBuilder();
        while(tokenizer.hasMoreElements()) {
            sb.append(tokenizer.nextToken());
        }
        return sb.toString();
    }

    public static String bestMove(){
        return new Message(Type.BEST_MOVE).toString();
    }

    public static String iPlay(String move){
        return new Message(Type.I_PLAY, move).toString();
    }

    public static String otherPlay(String move){
        return new Message(Type.OTHER_PLAY, move).toString();
    }

    public static String role(String role){
        return new Message(Type.ROLE, role).toString();
    }

    public static String teamName(){
        return new Message(Type.TEAM_NAME).toString();
    }

    public static String victory(){
        return new Message(Type.VICTORY).toString();
    }

    public static String defeat(){
        return new Message(Type.DEFEAT).toString();
    }

    public static String tie() {
        return new Message(Type.TIE).toString();
    }

    // public static String timeout() {
    //     return new Message(Type.TIMEOUT).toString();
    // }


    @Override
    public String toString() {
        return type + SEP + msg + END;
    }

}
